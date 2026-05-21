"""High-performance Medical OCR Flask Server.

Based on the original server.py with enhanced concurrency support:
- Thread pool for parallel request processing
- Redis-based caching for duplicate requests
- Prometheus metrics for monitoring
- Batch processing support
- Graceful shutdown handling
"""

import os
import sys
import time
import traceback
import uuid
import multiprocessing
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List
from functools import lru_cache

try:
    from flask import Flask, request, jsonify, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    request = None
    jsonify = None
    Response = None

from glmocr.config import load_config, GlmOcrConfig
from glmocr.utils.logging import get_logger, configure_logging

from medical_ocr.pipeline import MedicalOcrPipeline
from medical_ocr.layout_detector import MedicalLayoutDetector

logger = get_logger(__name__)

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# =========================================================================
# High-Performance Configuration
# =========================================================================

class HighPerfConfig:
    """High-performance server configuration."""
    
    # Thread pool settings
    THREAD_POOL_SIZE: int = int(os.getenv("THREAD_POOL_SIZE", "8"))
    
    # Redis cache settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", "3600"))
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    
    # Batch processing settings
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "50"))
    
    # Prometheus settings
    PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "false").lower() == "true"
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "8001"))

# Global instances
high_perf_config = HighPerfConfig()
executor = None
redis_client = None

# =========================================================================
# Prometheus Metrics (if enabled)
# =========================================================================

if HighPerfConfig.PROMETHEUS_ENABLED:
    try:
        from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
        METRICS_ENABLED = True
        
        REQUEST_COUNT = Counter(
            "ocr_requests_total",
            "Total OCR requests",
            ["endpoint", "status"]
        )
        REQUEST_LATENCY = Histogram(
            "ocr_request_duration_seconds",
            "OCR request duration in seconds",
            ["endpoint"]
        )
        CACHE_HITS = Counter("ocr_cache_hits_total", "Number of cache hits")
        CACHE_MISSES = Counter("ocr_cache_misses_total", "Number of cache misses")
        ACTIVE_WORKERS = Gauge("ocr_active_workers", "Number of active workers")
        PENDING_TASKS = Gauge("ocr_pending_tasks", "Number of pending tasks")
        
    except ImportError:
        METRICS_ENABLED = False
        logger.warning("prometheus_client not installed, metrics disabled")
else:
    METRICS_ENABLED = False

# =========================================================================
# Redis Cache Manager
# =========================================================================

class RedisCacheManager:
    """Redis-based cache manager for OCR results."""
    
    def __init__(self, host: str, port: int, db: int, ttl: int):
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl
        self._client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis."""
        try:
            import redis
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self._client.ping()
            logger.info(f"Redis connected: {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self._client = None
    
    def _generate_key(self, data: Dict) -> str:
        """Generate cache key from request data."""
        serialized = json.dumps(data, sort_keys=True)
        return f"ocr:cache:{hashlib.md5(serialized.encode()).hexdigest()}"
    
    def get(self, data: Dict) -> Optional[Dict]:
        """Get cached result."""
        if not self._client:
            return None
        
        try:
            key = self._generate_key(data)
            value = self._client.get(key)
            
            if METRICS_ENABLED:
                if value:
                    CACHE_HITS.inc()
                else:
                    CACHE_MISSES.inc()
            
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    def set(self, data: Dict, result: Dict):
        """Set cache with TTL."""
        if not self._client:
            return
        
        try:
            key = self._generate_key(data)
            self._client.setex(key, self.ttl, json.dumps(result))
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    def clear(self):
        """Clear all OCR cache."""
        if not self._client:
            return
        
        try:
            keys = self._client.keys("ocr:cache:*")
            if keys:
                self._client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries")
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        if not self._client:
            return {"enabled": False}
        
        try:
            keys = self._client.keys("ocr:cache:*")
            return {
                "enabled": True,
                "cached_items": len(keys),
                "ttl_seconds": self.ttl
            }
        except Exception:
            return {"enabled": False}

# =========================================================================
# High-Performance Request Processor
# =========================================================================

class HighPerfProcessor:
    """Process OCR requests with thread pool and caching."""
    
    def __init__(self, pipeline: MedicalOcrPipeline, cache_manager: RedisCacheManager = None):
        self.pipeline = pipeline
        self.cache = cache_manager
        self._lock = threading.Lock()
    
    def _build_request_data(self, images: List[str]) -> Dict:
        """Build request data for pipeline."""
        messages = [{"role": "user", "content": []}]
        for image_url in images:
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )
        return {"messages": messages}
    
    def process_single(
        self, 
        images: List[str],
        preprocess_context: Optional[Dict] = None,
        postprocess_context: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict:
        """Process a single OCR request."""
        start_time = time.time()
        request_data = self._build_request_data(images)
        
        # Check cache first
        if use_cache and self.cache:
            cached_result = self.cache.get(request_data)
            if cached_result:
                cached_result["cached"] = True
                cached_result["processing_time"] = time.time() - start_time
                return cached_result
        
        # Process request
        try:
            results = list(
                self.pipeline.process(
                    request_data,
                    save_layout_visualization=False,
                    preprocess_context=preprocess_context or {},
                    postprocess_context=postprocess_context or {},
                )
            )
            
            # Build response
            if not results:
                result = {
                    "json_result": None,
                    "markdown_result": "",
                    "pages": 0
                }
            elif len(results) == 1:
                result = {
                    "json_result": results[0].json_result,
                    "markdown_result": results[0].markdown_result,
                    "pages": 1
                }
            else:
                result = {
                    "json_result": [r.json_result for r in results],
                    "markdown_result": "\n\n---\n\n".join(r.markdown_result or "" for r in results),
                    "pages": len(results)
                }
            
            # Cache the result
            if use_cache and self.cache:
                self.cache.set(request_data, result)
            
            result["cached"] = False
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Process error: {e}")
            raise
    
    def process_batch(self, requests: List[Dict]) -> List[Dict]:
        """Process batch OCR requests using thread pool."""
        if METRICS_ENABLED:
            PENDING_TASKS.set(len(requests))
        
        results = []
        futures = []
        
        # Submit all tasks
        for idx, req in enumerate(requests):
            images = req.get("images", [])
            if isinstance(images, str):
                images = [images]
            
            future = executor.submit(
                self.process_single,
                images,
                req.get("preprocess_options"),
                req.get("postprocess_options"),
                True
            )
            futures.append((idx, future))
        
        # Collect results
        for idx, future in futures:
            try:
                result = future.result(timeout=120)
                results.append({
                    "index": idx,
                    "success": True,
                    "data": result
                })
            except Exception as e:
                results.append({
                    "index": idx,
                    "success": False,
                    "error": str(e)
                })
        
        if METRICS_ENABLED:
            PENDING_TASKS.set(0)
        
        return results

# =========================================================================
# Response Builder (from original server.py)
# =========================================================================

def _build_response(
    json_result: Optional[str],
    markdown_result: Optional[str],
    extra_info: Optional[Dict[str, Any]] = None,
    cached: bool = False,
    processing_time: float = 0.0
) -> Dict[str, Any]:
    """Build API response with medical OCR specific metadata."""
    response = {
        # SDK native fields
        "json_result": json_result,
        "markdown_result": markdown_result,
        # MaaS-compatible fields
        "layout_details": json_result,
        "md_results": markdown_result,
        "data_info": {"pages": []},
        "usage": {},
        "model": "medical-ocr",
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "created": int(time.time()),
        # Medical OCR specific
        "is_medical_ocr": True,
        "version": "0.1.0",
        # High-performance metadata
        "cached": cached,
        "processing_time": processing_time,
    }
    
    if extra_info:
        response.update(extra_info)
        
    return response

# =========================================================================
# Flask Application (Enhanced from original)
# =========================================================================

def create_app(config: GlmOcrConfig) -> Flask:
    """Create High-Performance Medical OCR Flask application.
    
    Args:
        config: Configuration object
        
    Returns:
        Flask application instance with high-performance features
    """
    if not FLASK_AVAILABLE:
        raise ImportError(
            "Flask server support requires the optional server extra. "
            "Install with: pip install flask"
        )

    app = Flask(__name__)
    
    # Initialize cache manager
    global redis_client
    if high_perf_config.REDIS_ENABLED:
        redis_client = RedisCacheManager(
            host=high_perf_config.REDIS_HOST,
            port=high_perf_config.REDIS_PORT,
            db=high_perf_config.REDIS_DB,
            ttl=high_perf_config.REDIS_CACHE_TTL
        )

    # Initialize medical-specific layout detector
    medical_layout_detector = MedicalLayoutDetector(config.pipeline.layout)
    
    # Initialize medical OCR pipeline
    pipeline = MedicalOcrPipeline(
        config=config.pipeline,
        layout_detector=medical_layout_detector
    )

    # Initialize high-performance processor
    processor = HighPerfProcessor(pipeline, redis_client)

    # Store in app config
    app.config["pipeline"] = pipeline
    app.config["processor"] = processor
    app.config["doc_config"] = config
    app.config["is_medical_ocr"] = True
    app.config["redis_cache"] = redis_client

    # =====================================================================
    # Health Check Endpoints
    # =====================================================================

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "ok",
            "service": "medical-ocr",
            "version": "0.0.2",
            "is_medical_ocr": True,
            "high_performance": True,
            "thread_pool_size": high_perf_config.THREAD_POOL_SIZE,
            "redis_enabled": redis_client is not None,
            "prometheus_enabled": METRICS_ENABLED,
        })

    @app.route("/health/ready", methods=["GET"])
    def readiness_check():
        """Readiness check endpoint."""
        try:
            pipeline = app.config["pipeline"]
            return jsonify({
                "status": "ready",
                "pipeline_running": pipeline is not None
            })
        except Exception as e:
            return jsonify({
                "status": "not_ready",
                "error": str(e)
            }), 503

    # =====================================================================
    # Main OCR Endpoint (Enhanced with caching)
    # =====================================================================

    @app.route("/medical-ocr/parse", methods=["POST"])
    def parse_medical_document():
        """Medical document parsing endpoint with caching.
        
        Original logic preserved, only added caching layer.
        """
        if request.headers.get("Content-Type") != "application/json":
            return jsonify({
                "error": "Invalid Content-Type. Expected 'application/json'."
            }), 400

        try:
            data = request.json
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Get images from request (same as original)
        images = data.get("images", [])
        if isinstance(images, str):
            images = [images]

        if not images and "file" in data:
            file_val = data["file"]
            if isinstance(file_val, str) and file_val:
                images = [file_val]

        if not images:
            return jsonify({"error": "No images provided"}), 400

        # Get processing options (same as original)
        preprocess_context = data.get("preprocess_options", {})
        postprocess_context = data.get("postprocess_options", {})

        try:
            # Process with high-performance processor
            processor = app.config["processor"]
            result = processor.process_single(
                images,
                preprocess_context,
                postprocess_context,
                use_cache=True
            )

            return jsonify(_build_response(
                result["json_result"],
                result["markdown_result"],
                cached=result.get("cached", False),
                processing_time=result.get("processing_time", 0.0)
            )), 200

        except Exception as e:
            logger.error(f"Parse error: {e}")
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

    # =====================================================================
    # Enhanced OCR Endpoint (from original)
    # =====================================================================

    @app.route("/medical-ocr/parse/enhanced", methods=["POST"])
    def parse_enhanced_medical():
        """Enhanced medical document parsing with advanced options."""
        if request.headers.get("Content-Type") != "application/json":
            return jsonify({
                "error": "Invalid Content-Type. Expected 'application/json'."
            }), 400

        try:
            data = request.json
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
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

        medical_enhancements = data.get("medical_enhancements", {})
        postprocess_context = {
            "enhancements": medical_enhancements
        }

        try:
            processor = app.config["processor"]
            result = processor.process_single(
                images,
                None,
                postprocess_context,
                use_cache=True
            )

            extra_info = {
                "medical_enhancements_applied": medical_enhancements
            }

            return jsonify(_build_response(
                result["json_result"],
                result["markdown_result"],
                extra_info,
                cached=result.get("cached", False),
                processing_time=result.get("processing_time", 0.0)
            )), 200

        except Exception as e:
            logger.error(f"Enhanced parse error: {e}")
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

    # =====================================================================
    # Batch Processing Endpoint (NEW)
    # =====================================================================

    @app.route("/medical-ocr/batch", methods=["POST"])
    def batch_parse_medical():
        """Batch OCR processing endpoint.
        
        NEW: Process multiple OCR requests in parallel using thread pool.
        """
        if request.headers.get("Content-Type") != "application/json":
            return jsonify({
                "error": "Invalid Content-Type. Expected 'application/json'."
            }), 400

        try:
            data = request.json
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return jsonify({"error": "Invalid JSON payload"}), 400

        requests = data.get("requests", [])
        
        if not requests:
            return jsonify({"error": "No requests provided"}), 400
        
        if len(requests) > high_perf_config.MAX_BATCH_SIZE:
            return jsonify({
                "error": f"Batch size exceeds maximum of {high_perf_config.MAX_BATCH_SIZE}"
            }), 400

        start_time = time.time()

        try:
            processor = app.config["processor"]
            results = processor.process_batch(requests)

            total_time = time.time() - start_time
            success_count = sum(1 for r in results if r["success"])
            failed_count = len(results) - success_count

            return jsonify({
                "results": results,
                "total": len(results),
                "success_count": success_count,
                "failed_count": failed_count,
                "processing_time": total_time
            }), 200

        except Exception as e:
            logger.error(f"Batch parse error: {e}")
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Batch parse error: {str(e)}"}), 500

    # =====================================================================
    # Cache Management Endpoints (NEW)
    # =====================================================================

    @app.route("/cache/stats", methods=["GET"])
    def cache_stats():
        """Get cache statistics."""
        cache = app.config.get("redis_cache")
        
        if not cache:
            return jsonify({
                "enabled": False,
                "message": "Redis cache is not enabled"
            })
        
        stats = cache.get_stats()
        return jsonify(stats)

    @app.route("/cache", methods=["DELETE"])
    def clear_cache():
        """Clear all cached OCR results."""
        cache = app.config.get("redis_cache")
        
        if not cache:
            return jsonify({
                "success": False,
                "message": "Redis cache is not enabled"
            }), 400
        
        cache.clear()
        return jsonify({
            "success": True,
            "message": "Cache cleared successfully"
        })

    # =====================================================================
    # Prometheus Metrics Endpoint (NEW)
    # =====================================================================

    if METRICS_ENABLED:
        @app.route("/metrics", methods=["GET"])
        def metrics():
            """Prometheus metrics endpoint."""
            return Response(
                generate_latest(),
                mimetype=CONTENT_TYPE_LATEST
            )

    return app


# =========================================================================
# Main Entry Point
# =========================================================================

def main():
    """Main entry point for High-Performance Medical OCR server."""
    import argparse

    parser = argparse.ArgumentParser(description="High-Performance Medical OCR Server")
    parser.add_argument(
        "--config", type=str, default=None, help="Config file path"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to listen on"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )
    parser.add_argument(
        "--redis-enabled", action="store_true", default=False,
        help="Enable Redis caching"
    )
    parser.add_argument(
        "--redis-host", type=str, default="localhost",
        help="Redis host"
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379,
        help="Redis port"
    )
    
    args = parser.parse_args()

    # Set multiprocessing start method
    multiprocessing.set_start_method("spawn", force=True)

    # Initialize global executor
    global executor
    high_perf_config.THREAD_POOL_SIZE = args.workers
    high_perf_config.REDIS_ENABLED = args.redis_enabled
    high_perf_config.REDIS_HOST = args.redis_host
    high_perf_config.REDIS_PORT = args.redis_port
    
    executor = ThreadPoolExecutor(max_workers=high_perf_config.THREAD_POOL_SIZE)
    logger.info(f"Thread pool initialized with {high_perf_config.THREAD_POOL_SIZE} workers")

    app = None

    try:
        config = load_config(args.config)

        # Configure logging
        log_level = args.log_level or getattr(config.logging, "level", "INFO")
        configure_logging(level=log_level)

        # Create app
        app = create_app(config)

        # Start pipeline
        pipeline = app.config["pipeline"]
        pipeline.start()

        # Get server config
        server_config = getattr(config, "server", None)
        host = args.host or (server_config.host if server_config else "0.0.0.0")
        port = args.port or (server_config.port if server_config else 8080)
        debug = getattr(server_config, "debug", False) if server_config else False

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"High-Performance Medical OCR Server starting on {host}:{port}")
        logger.info("=" * 70)
        logger.info(f"Thread Pool Size: {high_perf_config.THREAD_POOL_SIZE}")
        logger.info(f"Redis Caching: {'Enabled' if high_perf_config.REDIS_ENABLED else 'Disabled'}")
        logger.info(f"Prometheus Metrics: {'Enabled' if METRICS_ENABLED else 'Disabled'}")
        logger.info("=" * 70)
        logger.info("Endpoints:")
        logger.info("  - GET  /health              : Health check")
        logger.info("  - POST /medical-ocr/parse    : Single OCR processing")
        logger.info("  - POST /medical-ocr/parse/enhanced : Enhanced OCR")
        logger.info("  - POST /medical-ocr/batch    : Batch OCR processing")
        logger.info("  - GET  /cache/stats         : Cache statistics")
        logger.info("  - DELETE /cache             : Clear cache")
        if METRICS_ENABLED:
            logger.info("  - GET  /metrics             : Prometheus metrics")
        logger.info("=" * 70)
        logger.info("")

        # Run with threaded mode for better concurrency
        app.run(
            host=host, 
            port=port, 
            debug=debug,
            threaded=True,
            use_reloader=False
        )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        if executor:
            executor.shutdown(wait=True)
        if app is not None and "pipeline" in app.config:
            try:
                app.config["pipeline"].stop()
            except Exception as e:
                logger.warning(f"Error stopping pipeline: {e}")


if __name__ == "__main__":
    main()
