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
import multiprocessing
from typing import Dict, Any, Optional, Tuple

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

from medical_ocr.pipeline_pool import create_pipeline_pool
from glmocr.utils.logging import get_logger, configure_logging

logger = get_logger(__name__)

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


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

    # Load config if not provided
    if config is None:
        from glmocr.config import load_config
        config = load_config()

    # Create pipeline pool
    pipeline_pool = create_pipeline_pool(
        pipeline_config=config.pipeline if hasattr(config, 'pipeline') else config,
        pool_size=pool_size,
        yolo_model_dir=yolo_model_dir,
        uvdoc_model_dir=uvdoc_model_dir,
        use_processes=use_processes,
        use_gpu=use_gpu,
        gpu_ids=gpu_ids,
    )

    # Create cache
    cache = SimpleCache()

    # Store in app config
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

    # Also read from environment variables (env vars override CLI defaults, CLI args override env)
    pool_size = int(os.getenv("POOL_SIZE", str(args.pool_size)))
    use_processes = (os.getenv("USE_PROCESSES", "true").lower() == "true") if os.getenv("USE_PROCESSES") else args.use_processes
    use_gpu = (os.getenv("USE_GPU", "true").lower() == "true") if os.getenv("USE_GPU") else args.use_gpu
    gpu_ids_str = os.getenv("GPU_IDS", args.gpu_ids)
    gpu_ids = [int(x) for x in gpu_ids_str.split(",")] if gpu_ids_str else None
    yolo_model_dir = os.getenv("YOLO_MODEL_DIR", args.yolo_model_dir)
    uvdoc_model_dir = os.getenv("UVDOC_MODEL_DIR", args.uvdoc_model_dir)

    multiprocessing.set_start_method("spawn", force=True)

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

        # Start pipeline pool
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