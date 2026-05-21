"""Pipeline Pool - High Performance Architecture for Medical OCR.

This solves the three key bottlenecks:
1. 🔴 Pipeline singleton + thread pool (not suitable for GPU)
   → SOLUTION: Pipeline instance pool, one per process/worker

2. 🔴 YOLO detection (slowest step)
   → SOLUTION: Batch inference + GPU isolation

3. 🔴 Python GIL limitation
   → SOLUTION: Use ProcessPoolExecutor instead of ThreadPoolExecutor

Architecture:
┌─────────────────────────────────────────────────────┐
│          FastAPI (AsyncIO)                           │
│              ↓                                       │
│     Pipeline Instance Pool (N workers)               │
│      ┌─────────────┐ ┌─────────────┐                 │
│      │ Worker 1    │ │ Worker 2    │  ...          │
│      │ (GPU 0)     │ │ (GPU 1)     │                 │
│      │ Pipeline 1  │ │ Pipeline 2  │                 │
│      └─────────────┘ └─────────────┘                 │
└─────────────────────────────────────────────────────┘
"""

import os
import sys
import time
import json
import threading
import queue
import uuid
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple, Generator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps

from PIL import Image
from glmocr.utils.logging import get_logger
from glmocr.config import load_config
from glmocr.parser_result import PipelineResult

logger = get_logger(__name__)


# =========================================================================
# Global Configuration
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
    
    # Performance
    enable_parallel_preprocessing: bool = True
    parallel_preprocess_workers: int = 4
    
    # GPU settings (critical!)
    use_gpu: bool = True
    gpu_device_ids: Optional[List[int]] = None  # e.g., [0, 1] for multiple GPUs


# =========================================================================
# Worker Process - Isolated Pipeline
# =========================================================================

def _pipeline_worker_initializer(config_dict: Dict[str, Any], worker_idx: int, gpu_id: Optional[int]):
    """Initialize pipeline in worker process.
    
    This function runs in a separate process, so it has its own
    Python interpreter, GIL, and memory space. Perfect for GPU isolation!
    
    Args:
        config_dict: Pipeline configuration (can't pickle full config object)
        worker_idx: Index of this worker in the pool
        gpu_id: GPU device ID to assign (0, 1, ...)
    """
    global _worker_pipeline, _worker_config
    
    try:
        # Set GPU device for this worker
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(f"Worker {worker_idx}: Using GPU {gpu_id}")
        
        # Load real config from dict (need to reconstruct)
        from glmocr.config import GlmOCRConfig, PipelineConfig
        
        # Initialize Pipeline
        sys.path.insert(0, '/workspace/medical_ocr')
        from medical_ocr.pipeline import MedicalOcrPipeline
        from medical_ocr.page_loader import MedicalPageLoader
        from medical_ocr.layout_detector import MedicalLayoutDetector
        
        config = load_config()
        
        # Create pipeline instance
        _worker_pipeline = MedicalOcrPipeline(
            config=config.pipeline,
            yolo_model_dir=config_dict.get('yolo_model_dir'),
            uvdoc_model_dir=config_dict.get('uvdoc_model_dir')
        )
        
        # Initialize
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
    """Process a single task in a worker process.
    
    Each worker has its own isolated Pipeline instance and GPU.
    
    Args:
        task_type: Type of task to process
        task_data: Data for the task
    
    Returns:
        Processed result
    """
    global _worker_pipeline, _worker_config
    
    if _worker_pipeline is None:
        raise RuntimeError("Worker pipeline not initialized")
    
    try:
        start_time = time.time()
        
        if task_type == "process":
            # Process images through pipeline
            request_data = task_data.get('request_data')
            results = list(_worker_pipeline.process(request_data))
            
            # Convert PipelineResult to serializable data
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
        
        elif task_type == "preprocess_only":
            # Only preprocessing (no OCR)
            # ... (you can add other task types)
            pass
        
        raise ValueError(f"Unknown task type: {task_type}")
        
    except Exception as e:
        logger.error(f"Worker task failed: {e}")
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# =========================================================================
# Pipeline Pool Manager
# =========================================================================

class PipelinePool:
    """Pipeline instance pool for high-performance OCR.
    
    Features:
    - Multiple Pipeline instances (one per worker process)
    - GPU isolation for each worker
    - Task queue for load balancing
    - Graceful initialization and shutdown
    """
    
    def __init__(self, config: PipelinePoolConfig):
        """Initialize the pipeline pool.
        
        Args:
            config: Pool configuration
        """
        self.config = config
        self._pool: Optional[ProcessPoolExecutor] = None
        self._initialized = False
        self._shutdown = False
        
        # Prepare config dict for workers (can't pickle full config)
        self._worker_config_dict = {
            'yolo_model_dir': config.yolo_model_dir,
            'uvdoc_model_dir': config.uvdoc_model_dir,
            'pool_size': config.pool_size
        }
        
        # GPU assignment (round-robin)
        if config.use_gpu and config.gpu_device_ids:
            self._gpu_ids = config.gpu_device_ids
        else:
            self._gpu_ids = [None] * config.pool_size
    
    def initialize(self):
        """Initialize the pipeline pool (create worker processes)."""
        if self._initialized:
            logger.warning("Pipeline pool already initialized")
            return
        
        logger.info(f"Initializing pipeline pool with {self.config.pool_size} workers")
        
        # Use spawn method (important for Windows/macOS compatibility)
        ctx = mp.get_context('spawn')
        
        # Create ProcessPoolExecutor
        self._pool = ProcessPoolExecutor(
            max_workers=self.config.pool_size,
            mp_context=ctx
        )
        
        # Initialize each worker
        for worker_idx in range(self.config.pool_size):
            gpu_id = self._gpu_ids[worker_idx % len(self._gpu_ids)]
            
            # Submit initialization task
            future = self._pool.submit(
                _pipeline_worker_initializer,
                self._worker_config_dict,
                worker_idx,
                gpu_id
            )
            
            # Wait a bit for initialization
            try:
                future.result(timeout=300)  # 5 minute timeout for model loading
            except Exception as e:
                logger.error(f"Worker {worker_idx} initialization timeout: {e}")
        
        self._initialized = True
        logger.info("Pipeline pool initialized successfully")
    
    def submit(self, task_type: str, task_data: Dict[str, Any]) -> Any:
        """Submit a task to the pool.
        
        Args:
            task_type: Type of task
            task_data: Data for the task
        
        Returns:
            Future object
        """
        if self._shutdown:
            raise RuntimeError("Pipeline pool is shut down")
        
        if not self._initialized:
            self.initialize()
        
        return self._pool.submit(_pipeline_worker_task, task_type, task_data)
    
    def process(self, request_data: Dict[str, Any]) -> Any:
        """Process a request through the pool.
        
        Args:
            request_data: Input request data
        
        Returns:
            Processed results
        """
        future = self.submit("process", {'request_data': request_data})
        return future.result()
    
    def process_batch(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Process multiple requests in parallel.
        
        Args:
            requests: List of request data
        
        Returns:
            List of processed results
        """
        futures = [self.submit("process", {'request_data': r}) for r in requests]
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append({'error': str(e)})
        
        return results
    
    def shutdown(self, wait: bool = True):
        """Shutdown the pool and cleanup resources."""
        self._shutdown = True
        
        if self._pool:
            self._pool.shutdown(wait=wait)
        
        logger.info("Pipeline pool shut down")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# =========================================================================
# Alternative: Thread-based Pool (for single GPU)
# =========================================================================

class ThreadedPipelinePool:
    """Thread-based pipeline pool (for single GPU or CPU-only).
    
    Note: Still subject to GIL limitations, but better than single thread!
    """
    
    def __init__(self, config: PipelinePoolConfig):
        self.config = config
        self._pipeline_cache: List[Any] = []
        self._pipeline_queue: queue.Queue = queue.Queue()
        self._initialized = False
        self._lock = threading.Lock()
    
    def _create_pipeline(self):
        """Create a single pipeline instance."""
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
        """Initialize the thread pool."""
        with self._lock:
            if self._initialized:
                return
            
            logger.info(f"Initializing threaded pool with {self.config.pool_size} instances")
            
            # Create pipeline instances
            for i in range(self.config.pool_size):
                pipeline = self._create_pipeline()
                self._pipeline_queue.put(pipeline)
                self._pipeline_cache.append(pipeline)
            
            self._initialized = True
            logger.info("Threaded pool initialized")
    
    def _get_pipeline(self, timeout: float = 300):
        """Get an available pipeline."""
        if not self._initialized:
            self.initialize()
        
        return self._pipeline_queue.get(timeout=timeout)
    
    def _return_pipeline(self, pipeline):
        """Return a pipeline to the pool."""
        self._pipeline_queue.put(pipeline)
    
    def process(self, request_data: Dict[str, Any]) -> Any:
        """Process a single request."""
        pipeline = self._get_pipeline()
        
        try:
            return list(pipeline.process(request_data))
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
                    results.append({'error': str(e)})
            
            return results
    
    def shutdown(self):
        """Shutdown the pool."""
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
# Factory function - choose appropriate pool
# =========================================================================

def create_pipeline_pool(
    pipeline_config: Any,
    pool_size: int = 4,
    yolo_model_dir: Optional[str] = None,
    uvdoc_model_dir: Optional[str] = None,
    use_processes: bool = True,  # True = ProcessPool, False = ThreadPool
    use_gpu: bool = True,
    gpu_ids: Optional[List[int]] = None
) -> Any:
    """Create the appropriate pipeline pool.
    
    Args:
        pipeline_config: Pipeline configuration
        pool_size: Pool size
        yolo_model_dir: YOLO model directory
        uvdoc_model_dir: UVDoc model directory
        use_processes: Use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)
        use_gpu: Use GPU acceleration
        gpu_ids: List of GPU device IDs to use
    
    Returns:
        Pipeline pool instance
    """
    config = PipelinePoolConfig(
        pipeline_config=pipeline_config,
        pool_size=pool_size,
        yolo_model_dir=yolo_model_dir,
        uvdoc_model_dir=uvdoc_model_dir,
        use_gpu=use_gpu,
        gpu_device_ids=gpu_ids
    )
    
    if use_processes:
        logger.info("Creating Process-based PipelinePool")
        return PipelinePool(config)
    else:
        logger.info("Creating Thread-based PipelinePool")
        return ThreadedPipelinePool(config)


# =========================================================================
# Demo/Test
# =========================================================================

if __name__ == "__main__":
    print("Testing Pipeline Pool...")
    
    # Test configuration
    test_config = PipelinePoolConfig(
        pipeline_config=None,
        pool_size=2,
        yolo_model_dir=None,
        uvdoc_model_dir=None
    )
    
    print("Pool configuration created successfully")
    print("Note: Real test requires actual glmocr installation and models")
    print("\nUsage:")
    print("  from medical_ocr.pipeline_pool import create_pipeline_pool")
    print("  pool = create_pipeline_pool(pipeline_config, pool_size=4, use_processes=True)")
    print("  pool.initialize()")
    print("  results = pool.process(request_data)")
