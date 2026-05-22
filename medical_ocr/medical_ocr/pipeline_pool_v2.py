"""Pipeline Pool - Single GPU & Multi-GPU Support.

This supports both:
1. Single GPU, multiple Pipeline instances (thread-based)
2. Multiple GPUs, one Pipeline per GPU (process-based)

Single GPU Strategy:
- Use ThreadPoolExecutor (not ProcessPoolExecutor)
- Multiple Pipeline instances share the same GPU
- Requires controlling GPU memory usage
- Improves throughput (not pure parallelism, but better utilization)
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
from functools import wraps

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
    
    # Single-GPU mode settings
    single_gpu_mode: bool = True  # True = single GPU, multiple pipelines
    gpu_device_id: int = 0         # GPU ID for single-GPU mode
    gpu_memory_fraction: Optional[float] = None  # e.g., 0.5 = use 50% GPU memory
    
    # Multi-GPU mode settings
    multi_gpu_mode: bool = False
    gpu_device_ids: Optional[List[int]] = None  # e.g., [0, 1] for multiple GPUs
    
    # Performance
    enable_parallel_preprocessing: bool = True
    parallel_preprocess_workers: int = 4


# =========================================================================
# Single GPU Pipeline Pool (Thread-based)
# =========================================================================

class SingleGPUPipelinePool:
    """Pipeline pool for single-GPU, multiple-pipeline scenario.
    
    Uses ThreadPoolExecutor (not ProcessPoolExecutor) because:
    - All threads share the same GPU
    - Less memory overhead than processes
    - Better for interleaving GPU tasks
    - Avoids GIL for GPU operations (GPU is separate from CPU)
    """
    
    def __init__(self, config: PipelinePoolConfig):
        self.config = config
        self._pipeline_cache: List[Any] = []
        self._pipeline_queue: queue.Queue = queue.Queue()
        self._initialized = False
        self._lock = threading.Lock()
        
        # Set GPU device for all workers
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_device_id)
        
        # Optional: Limit GPU memory fraction
        if config.gpu_memory_fraction is not None:
            self._limit_gpu_memory(config.gpu_memory_fraction)
    
    def _limit_gpu_memory(self, fraction: float):
        """Limit GPU memory usage (PyTorch/TensorFlow specific)."""
        try:
            import torch
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"Set GPU memory limit to {fraction * 100}%")
        except Exception as e:
            logger.warning(f"Failed to limit GPU memory: {e}")
    
    def _create_pipeline(self):
        """Create a single pipeline instance (runs on main thread)."""
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
        """Initialize the pipeline pool (thread-based)."""
        with self._lock:
            if self._initialized:
                return
            
            logger.info(f"Initializing single-GPU pool with {self.config.pool_size} instances")
            
            # Create pipeline instances one by one
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
            logger.info(f"Single-GPU pool initialized with {len(self._pipeline_cache)} pipelines")
    
    def _get_pipeline(self, timeout: float = 300):
        """Get an available pipeline from the pool."""
        if not self._initialized:
            self.initialize()
        
        return self._pipeline_queue.get(timeout=timeout)
    
    def _return_pipeline(self, pipeline):
        """Return a pipeline to the pool."""
        self._pipeline_queue.put(pipeline)
    
    def process(self, request_data: Dict[str, Any]) -> Any:
        """Process a single request (thread-safe)."""
        pipeline = self._get_pipeline()
        
        try:
            # Multiple threads share the same GPU, but only one uses it at a time
            # (GPU driver handles scheduling)
            start_time = time.time()
            results = list(pipeline.process(request_data))
            
            # Build result dict
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
        """Process multiple requests in parallel (thread-based)."""
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
        """Shutdown the pool and cleanup resources."""
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
# Multi-GPU Pipeline Pool (Process-based)
# =========================================================================

def _multi_gpu_worker_initializer(config_dict: Dict[str, Any], worker_idx: int, gpu_id: int):
    """Initialize pipeline in worker process for multi-GPU mode."""
    global _worker_pipeline, _worker_config
    
    try:
        # Set GPU device for this worker (each worker gets its own GPU)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"Multi-GPU Worker {worker_idx}: Using GPU {gpu_id}")
        
        # Load and create pipeline
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
        
        logger.info(f"Multi-GPU Worker {worker_idx} initialized successfully")
        
    except Exception as e:
        logger.error(f"Worker {worker_idx} initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


def _multi_gpu_worker_task(task_type: str, task_data: Dict[str, Any]) -> Any:
    """Process a task in multi-GPU mode."""
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


class MultiGPUPipelinePool:
    """Pipeline pool for multiple-GPU scenario.
    
    Uses ProcessPoolExecutor with one Pipeline per GPU.
    """
    
    def __init__(self, config: PipelinePoolConfig):
        self.config = config
        self._pool: Optional[ProcessPoolExecutor] = None
        self._initialized = False
        self._shutdown = False
        
        self._worker_config_dict = {
            'yolo_model_dir': config.yolo_model_dir,
            'uvdoc_model_dir': config.uvdoc_model_dir
        }
    
    def initialize(self):
        """Initialize the multi-GPU pool."""
        if self._initialized:
            logger.warning("Multi-GPU pool already initialized")
            return
        
        logger.info(f"Initializing multi-GPU pool with {self.config.pool_size} workers")
        
        # Determine GPU assignment (round-robin)
        gpu_ids = self.config.gpu_device_ids or [0]
        logger.info(f"Using GPU IDs: {gpu_ids}")
        
        # Use spawn method
        ctx = mp.get_context('spawn')
        
        self._pool = ProcessPoolExecutor(
            max_workers=self.config.pool_size,
            mp_context=ctx
        )
        
        # Initialize each worker
        for worker_idx in range(self.config.pool_size):
            gpu_id = gpu_ids[worker_idx % len(gpu_ids)]
            
            future = self._pool.submit(
                _multi_gpu_worker_initializer,
                self._worker_config_dict,
                worker_idx,
                gpu_id
            )
            
            try:
                future.result(timeout=300)
            except Exception as e:
                logger.error(f"Worker {worker_idx} initialization failed: {e}")
        
        self._initialized = True
        logger.info("Multi-GPU pool initialized successfully")
    
    def submit(self, task_type: str, task_data: Dict[str, Any]) -> Any:
        """Submit a task to the pool."""
        if self._shutdown:
            raise RuntimeError("Pool is shut down")
        
        if not self._initialized:
            self.initialize()
        
        return self._pool.submit(_multi_gpu_worker_task, task_type, task_data)
    
    def process(self, request_data: Dict[str, Any]) -> Any:
        """Process a single request."""
        future = self.submit("process", {'request_data': request_data})
        return future.result()
    
    def process_batch(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Process multiple requests in parallel."""
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
        """Shutdown the pool."""
        self._shutdown = True
        
        if self._pool:
            self._pool.shutdown(wait=wait)
        
        logger.info("Multi-GPU pool shut down")
    
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
    mode: str = "single_gpu",  # "single_gpu" or "multi_gpu"
    gpu_device_id: int = 0,
    gpu_device_ids: Optional[List[int]] = None,
    gpu_memory_fraction: Optional[float] = None
) -> Any:
    """Create the appropriate pipeline pool based on mode.
    
    Args:
        mode: "single_gpu" (multiple pipelines on one GPU) or "multi_gpu" (one per GPU)
        gpu_device_id: Single GPU ID for single_gpu mode
        gpu_device_ids: List of GPU IDs for multi_gpu mode
        gpu_memory_fraction: Optional GPU memory limit (0-1) for single_gpu mode
    """
    config = PipelinePoolConfig(
        pipeline_config=pipeline_config,
        pool_size=pool_size,
        yolo_model_dir=yolo_model_dir,
        uvdoc_model_dir=uvdoc_model_dir,
        single_gpu_mode=(mode == "single_gpu"),
        gpu_device_id=gpu_device_id,
        gpu_memory_fraction=gpu_memory_fraction,
        multi_gpu_mode=(mode == "multi_gpu"),
        gpu_device_ids=gpu_device_ids
    )
    
    if mode == "single_gpu":
        logger.info(f"Creating Single-GPU Pipeline Pool (GPU {gpu_device_id}, {pool_size} pipelines)")
        return SingleGPUPipelinePool(config)
    
    elif mode == "multi_gpu":
        logger.info(f"Creating Multi-GPU Pipeline Pool (GPUs {gpu_device_ids}, {pool_size} workers)")
        return MultiGPUPipelinePool(config)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'single_gpu' or 'multi_gpu'.")


# =========================================================================
# Demo/Test
# =========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Pipeline Pool - Single & Multi GPU Support")
    print("=" * 70)
    
    print("\nUsage:")
    print("  # Single GPU, 4 pipeline instances")
    print("  from medical_ocr.pipeline_pool_v2 import create_pipeline_pool")
    print("  pool = create_pipeline_pool(")
    print("      config.pipeline, pool_size=4, mode='single_gpu', gpu_device_id=0")
    print("  )")
    
    print("\n  # Multi GPU, 2 workers per GPU")
    print("  pool = create_pipeline_pool(")
    print("      config.pipeline, pool_size=4, mode='multi_gpu', gpu_device_ids=[0, 1]")
    print("  )")
    
    print("\n  # Limit GPU memory in single GPU mode")
    print("  pool = create_pipeline_pool(")
    print("      ..., gpu_memory_fraction=0.5")
    print("  )")
    print("\n")
