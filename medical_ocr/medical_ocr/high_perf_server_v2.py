"""High Performance Medical OCR Server - Optimized Version.

This solves the three key bottlenecks:
1. Pipeline singleton + thread pool -> PipelinePool
2. YOLO detection -> Batch inference + GPU isolation
3. Python GIL -> ProcessPoolExecutor

Architecture:
                    Flask (Threaded)
                         |
                  PipelinePool (N workers)
            +----------+----------+----------+
            | Worker 1 | Worker 2 | ... | Worker N |
            | (GPU 0)  | (GPU 1)  |     | (GPU N)  |
            | Pipeline | Pipeline |     | Pipeline |
            +----------+----------+----------+
"""

import os
import sys
import time
import uuid
import json
import hashlib
import threading
import queue
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

try:
    from flask import Flask, request, jsonify

    _FLASK_IMPORT_ERROR = None
except ImportError as e:
    Flask = None
    request = None
    jsonify = None
    _FLASK_IMPORT_ERROR = e

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from glmocr.utils.logging import get_logger, configure_logging

logger = get_logger(__name__)

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


# =========================================================================
# Pipeline Pool Configuration
# =========================================================================

@dataclass
class PipelinePoolConfig:
    """Configuration for the pipeline pool."""
    pipeline_config: Any

    pool_size: int = 4
    max_queue_size: int = 100

    yolo_model_dir: Optional[str] = None
    uvdoc_model_dir: Optional[str] = None

    use_gpu: bool = False
    gpu_device_id: int = 0
    gpu_device_ids: Optional[List[int]] = None
    gpu_memory_fraction: Optional[float] = None

    enable_parallel_preprocessing: bool = True
    parallel_preprocess_workers: int = 4


# =========================================================================
# Worker Helpers (Process-based, for Multi-GPU / CPU)
# =========================================================================

_worker_pipeline = None
_worker_config = None

def _pipeline_worker_initializer(config_dict: Dict[str, Any], worker_idx: int, gpu_id: Optional[int]):
    """Initialize pipeline in worker process.

    Each worker runs in an isolated process with its own memory and (optionally) GPU.
    """
    global _worker_pipeline, _worker_config

    try:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(f"Worker {worker_idx}: Using GPU {gpu_id}")

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from medical_ocr.pipeline import MedicalOcrPipeline

        from glmocr.config import load_config
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
    """Process a task in a worker process."""
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
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from medical_ocr.pipeline import MedicalOcrPipeline

        from glmocr.config import load_config
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


# =========================================================================
# Simple Cache
# =========================================================================

class SimpleCache:
    """Simple in-memory cache with LRU eviction."""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def _get_key(self, request_data: Any) -> str:
        data_str = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def get(self, request_data: Any) -> Optional[Any]:
        key = self._get_key(request_data)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key][1]
            self._misses += 1
            return None

    def set(self, request_data: Any, result: Any, ttl: int = 3600):
        key = self._get_key(request_data)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), result)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                'hits': self._hits,
                'misses': self._misses,
                'size': len(self._cache)
            }

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# =========================================================================
# Response Builder
# =========================================================================

def _build_response(
    json_result,
    markdown_result,
    cached: bool = False,
    processing_time: float = 0.0,
    worker_idx: Optional[int] = None,
    gpu_id: Optional[int] = None,
    request_id: Optional[str] = None,
):
    """Build response dict with medical OCR specific metadata."""
    return {
        "request_id": request_id or f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "json_result": json_result,
        "markdown_result": markdown_result,
        "processing_time": processing_time,
        "cached": cached,
        "worker_idx": worker_idx,
        "gpu_id": gpu_id,
        "is_medical_ocr": True,
        "version": "0.1.0",
        "created": int(time.time()),
    }


# =========================================================================
# Flask Application Factory
# =========================================================================

def create_app(config=None, pool_size=4, use_processes=True, use_gpu=True, gpu_ids=None,
               yolo_model_dir=None, uvdoc_model_dir=None):
    """Create a Flask app with PipelinePool.

    Args:
        config: GlmOcrConfig instance (loaded from file if None).
        pool_size: Number of pipeline workers in the pool.
        use_processes: Use ProcessPoolExecutor (True) or ThreadedPipelinePool (False).
        use_gpu: Enable GPU acceleration.
        gpu_ids: List of GPU device IDs for multi-GPU mode.
        yolo_model_dir: Path to YOLO model directory.
        uvdoc_model_dir: Path to UVDoc model directory.

    Returns:
        Flask app instance.
    """
    if Flask is None:
        raise ImportError(
            "Flask server support requires Flask. Install with: pip install flask"
        ) from _FLASK_IMPORT_ERROR

    app = Flask(__name__)

    if config is None:
        from glmocr.config import load_config
        config = load_config()

    pipeline_pool = create_pipeline_pool(
        pipeline_config=config.pipeline if hasattr(config, 'pipeline') else config,
        pool_size=pool_size,
        yolo_model_dir=yolo_model_dir,
        uvdoc_model_dir=uvdoc_model_dir,
        use_processes=use_processes,
        use_gpu=use_gpu,
        gpu_ids=gpu_ids,
    )

    cache = SimpleCache()

    app.config["pipeline_pool"] = pipeline_pool
    app.config["cache"] = cache
    app.config["pool_size"] = pool_size
    app.config["use_gpu"] = use_gpu

    # =====================================================================
    # Routes
    # =====================================================================

    @app.route("/", methods=["GET"])
    def root():
        return jsonify({
            "service": "medical-ocr-high-performance",
            "version": "0.1.0",
            "endpoints": {
                "health": "/health",
                "parse": "/ocr/parse",
                "batch": "/ocr/batch",
                "cache": "/cache/stats",
            }
        })

    @app.route("/health", methods=["GET"])
    def health():
        pool = app.config["pipeline_pool"]
        cache_stats = app.config["cache"].stats()
        return jsonify({
            "status": "healthy" if pool else "initializing",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pool_size": app.config["pool_size"],
            "gpu_enabled": app.config["use_gpu"],
            "cache_hits": cache_stats['hits'],
            "cache_misses": cache_stats['misses'],
        })

    @app.route("/ocr/parse", methods=["POST"])
    def parse_document():
        """Process a single OCR request.

        Request:
            {
                "images": ["url1", "url2", ...],
                "options": {...}
            }
        """
        if request.headers.get("Content-Type") != "application/json":
            return jsonify({"error": "Invalid Content-Type. Expected 'application/json'."}), 400

        try:
            data = request.json
        except Exception:
            return jsonify({"error": "Invalid JSON payload"}), 400

        images = data.get("images", [])
        if isinstance(images, str):
            images = [images]

        if not images and "file" in data:
            file_val = data["file"]
            if isinstance(file_val, str) and file_val:
                images = [file_val]

        if not images:
            return jsonify({"error": "No images provided"}), 400

        start_time = time.time()
        request_id = str(uuid.uuid4())

        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in images
                    ]
                }
            ]
        }

        cache = app.config["cache"]

        cached_result = cache.get(request_data)
        if cached_result is not None:
            return jsonify(_build_response(
                cached_result.get('json_result'),
                cached_result.get('markdown_result'),
                cached=True,
                processing_time=time.time() - start_time,
                request_id=request_id,
            )), 200

        pool = app.config["pipeline_pool"]
        if pool is None:
            return jsonify({"error": "Server not ready"}), 503

        try:
            result = pool.process(request_data)

            if isinstance(result, dict) and 'error' in result:
                logger.error(f"Worker error: {result['error']}")
                return jsonify({"error": result['error']}), 500

            if result and isinstance(result, list) and len(result) > 0:
                single_result = result[0]
                cache.set(request_data, single_result)

                return jsonify(_build_response(
                    single_result.get('json_result'),
                    single_result.get('markdown_result'),
                    processing_time=time.time() - start_time,
                    worker_idx=single_result.get('worker_idx'),
                    gpu_id=single_result.get('gpu_id'),
                    request_id=request_id,
                )), 200

            return jsonify({"error": "Pipeline returned empty result"}), 500

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

    @app.route("/ocr/batch", methods=["POST"])
    def parse_batch():
        """Process batch OCR requests in parallel.

        Request:
            {
                "requests": [
                    {"images": ["url1", "url2", ...]},
                    {"images": ["url3", ...]},
                    ...
                ]
            }
        """
        if request.headers.get("Content-Type") != "application/json":
            return jsonify({"error": "Invalid Content-Type. Expected 'application/json'."}), 400

        try:
            data = request.json
        except Exception:
            return jsonify({"error": "Invalid JSON payload"}), 400

        batch_requests = data.get("requests", [])
        if not batch_requests:
            return jsonify({"error": "No requests provided"}), 400

        start_time = time.time()

        request_datas = []
        for req in batch_requests:
            images = req.get("images", [])
            if isinstance(images, str):
                images = [images]
            request_datas.append({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": url}}
                            for url in images
                        ]
                    }
                ]
            })

        pool = app.config["pipeline_pool"]
        if pool is None:
            return jsonify({"error": "Server not ready"}), 503

        try:
            results = pool.process_batch(request_datas)
            return jsonify({
                "batch_id": str(uuid.uuid4()),
                "results": results,
                "total_time": time.time() - start_time,
                "request_count": len(batch_requests),
            })

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/cache/stats", methods=["GET"])
    def cache_stats():
        return jsonify(app.config["cache"].stats())

    @app.route("/cache", methods=["DELETE"])
    def clear_cache():
        app.config["cache"].clear()
        return jsonify({"message": "Cache cleared successfully"})

    return app


# =========================================================================
# Main Entrypoint
# =========================================================================

def main():
    """Main entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Medical OCR High Performance Server")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--pool-size", type=int, default=4, help="Pipeline pool size")
    parser.add_argument(
        "--use-processes",
        action="store_true",
        default=True,
        help="Use ProcessPoolExecutor (multi-GPU). Disable for single GPU thread pool.",
    )
    parser.add_argument("--no-use-processes", action="store_false", dest="use_processes")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="Enable GPU")
    parser.add_argument("--no-use-gpu", action="store_false", dest="use_gpu")
    parser.add_argument(
        "--gpu-ids", type=str, default=None,
        help="Comma-separated GPU device IDs (e.g. '0,1,2,3')",
    )
    parser.add_argument("--yolo-model-dir", type=str, default=None, help="YOLO model dir")
    parser.add_argument("--uvdoc-model-dir", type=str, default=None, help="UVDoc model dir")
    args = parser.parse_args()

    pool_size = int(os.getenv("POOL_SIZE", str(args.pool_size)))
    use_processes = (os.getenv("USE_PROCESSES", "true").lower() == "true") if os.getenv("USE_PROCESSES") else args.use_processes
    use_gpu = (os.getenv("USE_GPU", "true").lower() == "true") if os.getenv("USE_GPU") else args.use_gpu
    gpu_ids_str = os.getenv("GPU_IDS", args.gpu_ids)
    gpu_ids = [int(x) for x in gpu_ids_str.split(",")] if gpu_ids_str else None
    yolo_model_dir = os.getenv("YOLO_MODEL_DIR", args.yolo_model_dir)
    uvdoc_model_dir = os.getenv("UVDOC_MODEL_DIR", args.uvdoc_model_dir)

    mp.set_start_method("spawn", force=True)

    app = None

    try:
        from glmocr.config import load_config

        config = load_config(args.config)

        log_level = args.log_level or config.logging.level
        configure_logging(level=log_level)

        app = create_app(
            config=config,
            pool_size=pool_size,
            use_processes=use_processes,
            use_gpu=use_gpu,
            gpu_ids=gpu_ids,
            yolo_model_dir=yolo_model_dir,
            uvdoc_model_dir=uvdoc_model_dir,
        )

        pipeline_pool = app.config["pipeline_pool"]
        pipeline_pool.initialize()

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Medical OCR High Performance Server starting on {args.host}:{args.port}")
        logger.info("=" * 70)
        logger.info(f"Pipeline Pool Size: {pool_size}")
        logger.info(f"Mode: {'Process (multi-GPU)' if use_processes else 'Thread (single GPU)'}")
        logger.info(f"GPU: {'Enabled' if use_gpu else 'Disabled'}")
        logger.info(f"GPU IDs: {gpu_ids}")
        logger.info("=" * 70)
        logger.info("Endpoints:")
        logger.info("  - GET  /health        : Health check")
        logger.info("  - POST /ocr/parse     : Single OCR processing")
        logger.info("  - POST /ocr/batch     : Batch OCR processing")
        logger.info("  - GET  /cache/stats   : Cache statistics")
        logger.info("  - DELETE /cache       : Clear cache")
        logger.info("=" * 70)
        logger.info("")

        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Error: %s", e)
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        if app is not None and "pipeline_pool" in app.config:
            try:
                app.config["pipeline_pool"].shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()