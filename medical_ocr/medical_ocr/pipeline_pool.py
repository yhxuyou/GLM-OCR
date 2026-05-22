"""Pipeline Pool - High Performance Architecture for Medical OCR.

Supports both single-GPU (thread-based) and multi-GPU (process-based) modes.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI / Flask (AsyncIO)                  │
│                      ↓                                     │
│              Pipeline Instance Pool (N workers)             │
│      ┌───────────────┐ ┌───────────────┐ ┌───────────────┐│
│      │  Worker 1     │ │  Worker 2     │ │  Worker N     ││
│      │  (GPU/Core 1) │ │  (GPU/Core 2) │ │  (GPU/Core N) ││
│      │  Pipeline 1   │ │  Pipeline 2   │ │  Pipeline N   ││
│      └───────────────┘ └───────────────┘ └───────────────┘│
└─────────────────────────────────────────────────────────────┘
"""

import os
import sys
import time
import json
import threading
import queue
import uuid
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from PIL import Image
from glmocr.utils.logging import get_logger
from glmocr.config import load_config
from glmocr.parser_result import PipelineResult

logger = get_logger(__name__)


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class PipelinePoolConfig:
    """Configuration for the pipeline pool."""
    # Pipeline configuration
    pipeline_config: Any

    # Pool configuration
    pool_size: int = 4
    max_queue_size: int = 100

    # Model paths
    yolo_model_dir: Optional[str] = None
    uvdoc_model_dir: Optional[str] = None

    # GPU settings
    use_gpu: bool = False
    gpu_device_id: int = 0
    gpu_device_ids: Optional[List[int]] = None
    gpu_memory_fraction: Optional[float] = None

    # Performance
    enable_parallel_preprocessing: bool = True
    parallel_preprocess_workers: int = 4


# =========================================================================
# Worker Helpers (Process-based, for Multi-GPU / CPU)
# =========================================================================

def _pipeline_worker_initializer(config_dict: Dict[str, Any], worker_idx: int, gpu_id: Optional[int]):
    """Initialize pipeline in worker process.

    Each worker runs in an isolated process with its own memory and (optionally) GPU.

    Args:
        config_dict: Pipeline configuration dictionary
        worker_idx: Index of this worker in the pool
        gpu_id: GPU device ID to assign (None = CPU)
    """
    global _worker_pipeline, _worker_config

    try:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(f"Worker {worker_idx}: Using GPU {gpu_id}")

        sys.path.insert(0, '/workspace/medical_ocr')
        from medical_ocr.pipeline import MedicalOcrPipeline

        config = load_config()
        _worker_pipeline = MedicalOcrPipeline(
            config=config.pipeline,
            yolo_model_dir=config_dict.get('yolo_model_dir'),
            uvdoc_model_dir=config_dict.get('uvdoc_model_dir')
        )
        _worker_pipeline.start()

        _worker_config = {
            'worker_idx': worker_idx,
            'gpu_id': gpu_id
        }

        logger.info(f"Worker {worker_idx} initialized successfully (GPU {gpu_id})")

    except Exception as e:
        logger.error(f"Worker {worker_idx} initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


def _pipeline_worker_task(task_type: str, task_data: Dict[str, Any]) -> Any:
    """Process a task in a worker process.

    Args:
        task_type: Type of task ("process")
        task_data: Data for the task

    Returns:
        Serialized result dict
    """
    global _worker_pipeline, _worker_config

    if _worker_pipeline is None:
        raise RuntimeError("Worker pipeline not initialized")

    try:
        start_time = time.time()

        if task_type == "process":
            request_data = task_data.get('request_data')
            results = list(_worker_pipeline.process(request_data))

            json_results = []
            for result in results:
                json_results.append({
                    'json_result': result.json_result,
                    'markdown_result': result.markdown_result,
                    'processing_time': time.time() - start_time,
                    'worker_idx': _worker_config.get('worker_idx'),
                    'gpu_id': _worker_config.get('gpu_id')
                })

            return json_results

        raise ValueError(f"Unknown task type: {task_type}")

    except Exception as e:
        logger.error(f"Worker task failed: {e}")
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# =========================================================================
# Process-based Pool (Multi-GPU / CPU)
# =========================================================================

class PipelinePool:
    """Process-based pipeline pool for high-performance OCR.

    Features:
    - Multiple Pipeline instances (one per worker process)
    - GPU isolation for each worker
    - Graceful initialization and shutdown
    """

    def __init__(self, config: PipelinePoolConfig):
        self.config = config
        self._pool: Optional[ProcessPoolExecutor] = None
        self._initialized = False
        self._shutdown = False

        self._worker_config_dict = {
            'yolo_model_dir': config.yolo_model_dir,
            'uvdoc_model_dir': config.uvdoc_model_dir,
            'pool_size': config.pool_size
        }

        if config.gpu_device_ids:
            self._gpu_ids = config.gpu_device_ids
        else:
            self._gpu_ids = [None] * config.pool_size

    def initialize(self):
        if self._initialized:
            logger.warning("Pipeline pool already initialized")
            return

        logger.info(f"Initializing process-based pool with {self.config.pool_size} workers")
        ctx = mp.get_context('spawn')

        self._pool = ProcessPoolExecutor(
            max_workers=self.config.pool_size,
            mp_context=ctx
        )

        for worker_idx in range(self.config.pool_size):
            gpu_id = self._gpu_ids[worker_idx % len(self._gpu_ids)]
            future = self._pool.submit(
                _pipeline_worker_initializer,
                self._worker_config_dict,
                worker_idx,
                gpu_id
            )
            try:
                future.result(timeout=300)
            except Exception as e:
                logger.error(f"Worker {worker_idx} initialization timeout: {e}")

        self._initialized = True
        logger.info("Process-based pool initialized successfully")

    def submit(self, task_type: str, task_data: Dict[str, Any]) -> Any:
        if self._shutdown:
            raise RuntimeError("Pipeline pool is shut down")
        if not self._initialized:
            self.initialize()
        return self._pool.submit(_pipeline_worker_task, task_type, task_data)

    def process(self, request_data: Dict[str, Any]) -> Any:
        future = self.submit("process", {'request_data': request_data})
        return future.result()

    def process_batch(self, requests: List[Dict[str, Any]]) -> List[Any]:
        futures = [self.submit("process", {'request_data': r}) for r in requests]
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append({'error': str(e)})
        return results

    def shutdown(self, wait: bool = True):
        self._shutdown = True
        if self._pool:
            self._pool.shutdown(wait=wait)
        logger.info("Pipeline pool shut down")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# =========================================================================
# Thread-based Pool (Single-GPU / CPU)
# =========================================================================

class ThreadedPipelinePool:
    """Thread-based pipeline pool (for single GPU or CPU-only).

    All Pipeline instances share the same GPU. Uses queue-based
    pipeline borrowing for thread-safe concurrent access.
    """

    def __init__(self, config: PipelinePoolConfig):
        self.config = config
        self._pipeline_cache: List[Any] = []
        self._pipeline_queue: queue.Queue = queue.Queue()
        self._initialized = False
        self._lock = threading.Lock()

        if config.gpu_device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_device_id)

        if config.gpu_memory_fraction is not None:
            self._limit_gpu_memory(config.gpu_memory_fraction)

    def _limit_gpu_memory(self, fraction: float):
        try:
            import torch
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"Set GPU memory limit to {fraction * 100}%")
        except Exception as e:
            logger.warning(f"Failed to limit GPU memory: {e}")

    def _create_pipeline(self):
        sys.path.insert(0, '/workspace/medical_ocr')
        from medical_ocr.pipeline import MedicalOcrPipeline

        config = load_config()
        pipeline = MedicalOcrPipeline(
            config=config.pipeline,
            yolo_model_dir=self.config.yolo_model_dir,
            uvdoc_model_dir=self.config.uvdoc_model_dir
        )
        pipeline.start()
        return pipeline

    def initialize(self):
        with self._lock:
            if self._initialized:
                return

            logger.info(f"Initializing thread-based pool with {self.config.pool_size} instances")

            for i in range(self.config.pool_size):
                logger.info(f"Creating pipeline instance {i+1}/{self.config.pool_size}")
                try:
                    pipeline = self._create_pipeline()
                    self._pipeline_queue.put(pipeline)
                    self._pipeline_cache.append(pipeline)
                    logger.info(f"Pipeline {i} created successfully")
                except Exception as e:
                    logger.error(f"Failed to create pipeline {i}: {e}")

            self._initialized = True
            logger.info(f"Thread-based pool initialized with {len(self._pipeline_cache)} pipelines")

    def _get_pipeline(self, timeout: float = 300):
        if not self._initialized:
            self.initialize()
        return self._pipeline_queue.get(timeout=timeout)

    def _return_pipeline(self, pipeline):
        self._pipeline_queue.put(pipeline)

    def process(self, request_data: Dict[str, Any]) -> Any:
        pipeline = self._get_pipeline()
        try:
            start_time = time.time()
            results = list(pipeline.process(request_data))

            json_results = []
            for result in results:
                json_results.append({
                    'json_result': result.json_result,
                    'markdown_result': result.markdown_result,
                    'processing_time': time.time() - start_time
                })
            return json_results
        finally:
            self._return_pipeline(pipeline)

    def process_batch(self, requests: List[Dict[str, Any]]) -> List[Any]:
        with ThreadPoolExecutor(max_workers=self.config.pool_size) as executor:
            futures = [executor.submit(self.process, r) for r in requests]
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    results.append({'error': str(e)})
            return results

    def shutdown(self):
        for pipeline in self._pipeline_cache:
            try:
                pipeline.stop()
            except Exception:
                pass

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# =========================================================================
# Factory Function - Smart Pool Selection
# =========================================================================

def create_pipeline_pool(
    pipeline_config: Any,
    pool_size: int = 4,
    yolo_model_dir: Optional[str] = None,
    uvdoc_model_dir: Optional[str] = None,
    use_processes: bool = True,
    use_gpu: bool = True,
    gpu_ids: Optional[List[int]] = None,
    mode: Optional[str] = None,
    gpu_device_id: int = 0,
    gpu_memory_fraction: Optional[float] = None
) -> Any:
    """Create the appropriate pipeline pool.

    Args:
        pipeline_config: Pipeline configuration
        pool_size: Pool size (default: 4)
        yolo_model_dir: YOLO model directory
        uvdoc_model_dir: UVDoc model directory
        use_processes: Use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)
        use_gpu: Use GPU acceleration
        gpu_ids: List of GPU device IDs (for multi-GPU mode)
        mode: "single_gpu" (thread-based) or "multi_gpu" (process-based, one per GPU).
              When set, overrides use_processes.
        gpu_device_id: Single GPU device ID (for single_gpu mode)
        gpu_memory_fraction: GPU memory limit (0-1, for single_gpu mode)

    Returns:
        Pipeline pool instance (PipelinePool or ThreadedPipelinePool)
    """
    config = PipelinePoolConfig(
        pipeline_config=pipeline_config,
        pool_size=pool_size,
        yolo_model_dir=yolo_model_dir,
        uvdoc_model_dir=uvdoc_model_dir,
        use_gpu=use_gpu,
        gpu_device_id=gpu_device_id,
        gpu_device_ids=gpu_ids,
        gpu_memory_fraction=gpu_memory_fraction
    )

    if mode == "single_gpu" or (mode is None and not use_processes):
        logger.info(f"Creating thread-based pool (GPU {gpu_device_id}, {pool_size} pipelines)")
        return ThreadedPipelinePool(config)

    if mode == "multi_gpu" or (mode is None and use_processes):
        logger.info(f"Creating process-based pool ({pool_size} workers, GPUs: {gpu_ids})")
        return PipelinePool(config)

    raise ValueError(f"Unknown mode: {mode}")