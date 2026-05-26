"""GLM-OCR Queue-based High-Concurrency Server.

Architecture:
    request → enqueue job → return job_id immediately
             ↓
    background workers (controlled concurrency via semaphore)
             ↓
    pipeline.process() → VLM backend

Key features:
    - Job queue decouples HTTP accept from pipeline processing
    - Semaphore limits concurrent pipeline executions
    - VLM rate limiter prevents backend overload
    - In-memory job store with TTL cleanup
    - Graceful shutdown with inflight-job draining
    - Health check and stats endpoints
    - Backward-compatible API with original server.py

Usage:
    python -m glmocr.server_queue --config config.yaml
    # or with gunicorn:
    gunicorn -w 8 -k sync glmocr.server_queue:build_wsgi_app
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import threading
import queue
import signal
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from flask import Flask, request, jsonify

    _FLASK_IMPORT_ERROR = None
except ImportError as e:
    Flask = None  # type: ignore
    request = None  # type: ignore
    jsonify = None  # type: ignore
    _FLASK_IMPORT_ERROR = e

from glmocr.pipeline import Pipeline
from glmocr.config import GlmOcrConfig, load_config
from glmocr.utils.logging import get_logger, configure_logging

try:
    from preprocess_client import PreprocessClient

    _PREPROCESS_CLIENT_IMPORT_ERROR = None
except ImportError as e:
    PreprocessClient = None  # type: ignore
    _PREPROCESS_CLIENT_IMPORT_ERROR = e

logger = get_logger(__name__)

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    images: List[str]
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    json_result: Optional[Any] = None
    markdown_result: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    preprocessed_images: List[str] = field(default_factory=list)
    preprocess_metadata: Optional[Dict[str, Any]] = None


def _build_response(json_result, markdown_result):
    return {
        "json_result": json_result,
        "markdown_result": markdown_result,
        "layout_details": json_result,
        "md_results": markdown_result,
        "data_info": {"pages": []},
        "usage": {},
        "model": "glm-ocr",
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "created": int(time.time()),
    }


class VlmRateLimiter:
    """Token-bucket rate limiter for VLM backend requests.

    Prevents the VLM backend from being overwhelmed by too many
    concurrent recognition requests.  Thread-safe.
    """

    def __init__(self, max_requests_per_second: float = 50.0):
        self._rate = max_requests_per_second
        self._tokens = float(max_requests_per_second)
        self._max_tokens = float(max_requests_per_second)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            time.sleep(0.01)


class JobStore:
    """Thread-safe in-memory job store with TTL-based cleanup.

    Stores job metadata and results.  Completed/failed jobs are
    automatically removed after *ttl_seconds*.
    """

    def __init__(self, ttl_seconds: int = 1800):
        self._store: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()

    def start_cleanup(self) -> None:
        if self._cleanup_thread is not None:
            return
        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="jobstore-cleanup"
        )
        self._cleanup_thread.start()

    def stop_cleanup(self) -> None:
        self._stop_cleanup.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=5)

    def _cleanup_loop(self) -> None:
        while not self._stop_cleanup.wait(60):
            self._purge_expired()

    def _purge_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                jid
                for jid, job in self._store.items()
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
                and (job.completed_at and (now - job.completed_at) > self._ttl)
            ]
            for jid in expired:
                del self._store[jid]
            if expired:
                logger.debug("Purged %d expired jobs", len(expired))

    def create(self, images: List[str]) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id, images=images)
        with self._lock:
            self._store[job_id] = job
        logger.debug("Job created: %s (%d images)", job_id, len(images))
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._store.get(job_id)

    def update_status(self, job_id: str, status: JobStatus, **kwargs: Any) -> None:
        with self._lock:
            job = self._store.get(job_id)
            if job is not None:
                job.status = status
                for k, v in kwargs.items():
                    setattr(job, k, v)
                if status == JobStatus.PROCESSING:
                    job.started_at = time.time()
                elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    job.completed_at = time.time()
                    job.progress = 100.0 if status == JobStatus.COMPLETED else 0.0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            counts = {s.value: 0 for s in JobStatus}
            for job in self._store.values():
                counts[job.status.value] += 1
            counts["total"] = len(self._store)
            return counts


class PipelineWorkerPool:
    """Background worker pool that pulls jobs from a queue and processes
    them through the OCR pipeline with controlled concurrency.

    Concurrency is limited by a semaphore so the VLM backend never sees
    more than *max_concurrent* pipeline executions at once.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        job_store: JobStore,
        max_concurrent: int = 3,
        max_queue_size: int = 500,
        vlm_rate_limit: Optional[float] = None,
        save_layout_visualization: bool = False,
        preprocess_enabled: bool = False,
        preprocess_url: str = "http://localhost:7001",
    ):
        self._pipeline = pipeline
        self._job_store = job_store
        self._max_concurrent = max_concurrent
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._job_queue: queue.Queue[Job] = queue.Queue(maxsize=max_queue_size)
        self._save_layout_viz = save_layout_visualization

        self._vlm_limiter: Optional[VlmRateLimiter] = None
        if vlm_rate_limit is not None and vlm_rate_limit > 0:
            self._vlm_limiter = VlmRateLimiter(vlm_rate_limit)

        self._preprocess_enabled = preprocess_enabled
        self._preprocess_url = preprocess_url
        self._preprocess_client: Optional[PreprocessClient] = None
        if preprocess_enabled and PreprocessClient:
            self._preprocess_client = PreprocessClient(preprocess_url)

        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="pipeline-wkr"
        )
        self._running = False
        self._stop_event = threading.Event()
        self._dispatcher: Optional[threading.Thread] = None
        self._stats: Dict[str, Any] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "queue_size": 0,
            "queue_maxsize": max_queue_size,
            "active_workers": 0,
            "max_workers": max_concurrent,
        }
        self._stats_lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="queue-dispatcher"
        )
        self._dispatcher.start()
        logger.info(
            "WorkerPool started: max_concurrent=%d, queue_maxsize=%d, vlm_rate_limit=%s",
            self._max_concurrent,
            self._job_queue.maxsize,
            self._vlm_limiter._rate if self._vlm_limiter else "unlimited",
        )

    def stop(self, timeout: float = 60.0) -> None:
        if not self._running:
            return
        logger.info("WorkerPool stopping (timeout=%ss)...", timeout)
        self._running = False
        self._stop_event.set()
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=5)
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.info("WorkerPool stopped")

    def submit(self, job: Job) -> bool:
        with self._stats_lock:
            self._stats["submitted"] += 1
        try:
            self._job_queue.put(job, timeout=1.0)
            self._update_queue_size()
            return True
        except queue.Full:
            with self._stats_lock:
                self._stats["submitted"] -= 1
            return False

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            s = dict(self._stats)
        s["queue_size"] = self._job_queue.qsize()
        return s

    def _update_queue_size(self) -> None:
        with self._stats_lock:
            self._stats["queue_size"] = self._job_queue.qsize()

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._job_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._semaphore.acquire()
            self._update_queue_size()
            with self._stats_lock:
                self._stats["active_workers"] = (
                    self._max_concurrent - self._semaphore._value
                )

            self._job_store.update_status(job.job_id, JobStatus.PROCESSING)
            logger.info(
                "Job %s started (queue_depth=%d, active=%d)",
                job.job_id,
                self._job_queue.qsize(),
                self._max_concurrent - self._semaphore._value,
            )

            future = self._executor.submit(self._process_job, job)
            future.add_done_callback(
                lambda f, j=job: self._on_job_done(j, f)
            )

    def _process_job(self, job: Job) -> None:
        if self._vlm_limiter is not None:
            self._vlm_limiter.acquire()

        images_to_process = job.images.copy()
        preprocess_metadata = []

        if self._preprocess_enabled and self._preprocess_client:
            try:
                preprocessed = []
                for img_url in job.images:
                    try:
                        result = self._preprocess_client.preprocess(img_url, sync=True)
                        if result.get("document_detected", False) and "corrected_image" in result:
                            preprocessed.append(result["corrected_image"])
                            preprocess_metadata.append({
                                "original_image": img_url,
                                "orientation_angle": result.get("orientation_angle", 0),
                                "deskewed_angle": result.get("deskewed_angle", 0.0),
                                "confidence": result.get("confidence", 0.0),
                                "document_detected": True,
                            })
                        else:
                            preprocessed.append(img_url)
                            preprocess_metadata.append({
                                "original_image": img_url,
                                "document_detected": False,
                                "warning": "Document not detected, using original image",
                            })
                    except Exception as e:
                        logger.warning(f"Preprocessing failed for {img_url}: {e}")
                        preprocessed.append(img_url)
                        preprocess_metadata.append({
                            "original_image": img_url,
                            "document_detected": False,
                            "warning": f"Preprocessing error: {str(e)}",
                        })
                images_to_process = preprocessed
                self._job_store.update_status(
                    job.job_id,
                    JobStatus.PROCESSING,
                    preprocessed_images=images_to_process,
                    preprocess_metadata=preprocess_metadata,
                )
                logger.info(f"Job {job.job_id}: preprocessed {len(images_to_process)} images")
            except Exception as e:
                logger.error(f"Preprocessing pipeline error: {e}")

        messages = [{"role": "user", "content": []}]
        for image_url in images_to_process:
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )
        request_data = {"messages": messages}

        results = list(
            self._pipeline.process(
                request_data,
                save_layout_visualization=self._save_layout_viz,
            )
        )

        if not results:
            self._job_store.update_status(
                job.job_id,
                JobStatus.COMPLETED,
                json_result=None,
                markdown_result="",
            )
            return

        if len(results) == 1:
            r = results[0]
            self._job_store.update_status(
                job.job_id,
                JobStatus.COMPLETED,
                json_result=r.json_result,
                markdown_result=r.markdown_result or "",
            )
        else:
            json_result = [r.json_result for r in results]
            markdown_result = "\n\n---\n\n".join(
                r.markdown_result or "" for r in results
            )
            self._job_store.update_status(
                job.job_id,
                JobStatus.COMPLETED,
                json_result=json_result,
                markdown_result=markdown_result,
            )

    def _on_job_done(self, job: Job, future: Future) -> None:
        self._semaphore.release()
        with self._stats_lock:
            self._stats["active_workers"] = (
                self._max_concurrent - self._semaphore._value
            )
        try:
            future.result()
        except Exception as e:
            logger.error("Job %s failed: %s", job.job_id, e)
            logger.debug(traceback.format_exc())
            actual_job = self._job_store.get(job.job_id)
            if actual_job and actual_job.status != JobStatus.COMPLETED:
                self._job_store.update_status(
                    job.job_id,
                    JobStatus.FAILED,
                    error_message=str(e),
                )


def create_queue_app(
    config: "GlmOcrConfig",
    job_store: Optional[JobStore] = None,
    worker_pool: Optional[PipelineWorkerPool] = None,
) -> Flask:
    if Flask is None:
        raise ImportError(
            "Flask server support requires the optional server extra. "
            "Install with: pip install 'glmocr[server]'"
        ) from _FLASK_IMPORT_ERROR

    app = Flask(__name__)

    pipeline = Pipeline(config=config.pipeline)

    queue_cfg = getattr(config, "queue", None)

    _job_store = job_store or JobStore(
        ttl_seconds=getattr(queue_cfg, "job_ttl_seconds", 1800)
        if queue_cfg
        else 1800
    )

    _worker_pool = worker_pool or PipelineWorkerPool(
        pipeline=pipeline,
        job_store=_job_store,
        max_concurrent=getattr(queue_cfg, "max_concurrent_pipelines", 3) if queue_cfg else 3,
        max_queue_size=getattr(queue_cfg, "max_queue_size", 500) if queue_cfg else 500,
        vlm_rate_limit=getattr(queue_cfg, "vlm_max_requests_per_second", None) if queue_cfg else None,
        save_layout_visualization=getattr(queue_cfg, "save_layout_visualization", False) if queue_cfg else False,
        preprocess_enabled=getattr(queue_cfg, "preprocess_enabled", False) if queue_cfg else False,
        preprocess_url=getattr(queue_cfg, "preprocess_url", "http://localhost:7001") if queue_cfg else "http://localhost:7001",
    )

    app.config["pipeline"] = pipeline
    app.config["doc_config"] = config
    app.config["job_store"] = _job_store
    app.config["worker_pool"] = _worker_pool

    @app.route("/glmocr/parse", methods=["POST"])
    def parse():
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

        job = _job_store.create(images)
        accepted = _worker_pool.submit(job)

        if not accepted:
            _job_store.update_status(
                job.job_id, JobStatus.FAILED, error_message="Queue is full, please retry later"
            )
            return (
                jsonify({
                    "error": "Queue is full",
                    "job_id": job.job_id,
                    "retry_after_seconds": 5,
                }),
                503,
            )

        return (
            jsonify({
                "job_id": job.job_id,
                "status": job.status.value,
                "created_at": job.created_at,
            }),
            202,
        )

    @app.route("/glmocr/status/<job_id>", methods=["GET"])
    def job_status(job_id: str):
        job = _job_store.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404

        return (
            jsonify({
                "job_id": job.job_id,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
            }),
            200,
        )

    @app.route("/glmocr/result/<job_id>", methods=["GET"])
    def job_result(job_id: str):
        job = _job_store.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404

        if job.status == JobStatus.PENDING:
            return (
                jsonify({"job_id": job_id, "status": "pending", "message": "Job is still queued"}),
                202,
            )
        if job.status == JobStatus.PROCESSING:
            return (
                jsonify({
                    "job_id": job_id,
                    "status": "processing",
                    "progress": job.progress,
                    "message": "Job is still processing",
                }),
                202,
            )
        if job.status == JobStatus.FAILED:
            return (
                jsonify({"job_id": job_id, "status": "failed", "error": job.error_message}),
                500,
            )

        return (
            jsonify(_build_response(job.json_result, job.markdown_result)),
            200,
        )

    @app.route("/glmocr/parse-sync", methods=["POST"])
    def parse_sync():
        """Synchronous endpoint (backward-compatible with original API).

        Blocks until the pipeline finishes.  Use for debugging or
        low-concurrency scenarios.
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

        messages = [{"role": "user", "content": []}]
        for image_url in images:
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )

        request_data = {"messages": messages}

        try:
            results = list(pipeline.process(request_data, save_layout_visualization=False))
            if not results:
                return jsonify(_build_response(None, "")), 200
            if len(results) == 1:
                r = results[0]
                return jsonify(_build_response(r.json_result, r.markdown_result or "")), 200
            json_result = [r.json_result for r in results]
            markdown_result = "\n\n---\n\n".join(r.markdown_result or "" for r in results)
            return jsonify(_build_response(json_result, markdown_result)), 200
        except Exception as e:
            logger.error("Parse error: %s", e)
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

    @app.route("/health", methods=["GET"])
    def health():
        stats = _worker_pool.get_stats()
        job_stats = _job_store.stats()
        return (
            jsonify({
                "status": "ok",
                "active_workers": stats["active_workers"],
                "max_workers": stats["max_workers"],
                "queue_size": stats["queue_size"],
                "queue_maxsize": stats["queue_maxsize"],
                "jobs": job_stats,
            }),
            200,
        )

    @app.route("/stats", methods=["GET"])
    def stats():
        stats = _worker_pool.get_stats()
        job_stats = _job_store.stats()
        return (
            jsonify({
                "pool": stats,
                "jobs": job_stats,
            }),
            200,
        )

    return app, _job_store, _worker_pool


def _ensure_queue_config(config: "GlmOcrConfig") -> "GlmOcrConfig":
    if getattr(config, "queue", None) is None:
        from glmocr.config import QueueConfig
        config.queue = QueueConfig()
    return config


# ── Module-level globals for lifecycle management ─────────────────
_app: Optional[Flask] = None
_job_store: Optional[JobStore] = None
_worker_pool: Optional[PipelineWorkerPool] = None


def _shutdown_handler(signum, frame):
    logger.info("Received signal %s, shutting down...", signum)
    if _worker_pool is not None:
        _worker_pool.stop(timeout=30)
    if _job_store is not None:
        _job_store.stop_cleanup()
    sys.exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GlmOcr Queue Server")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent pipeline executions (overrides config)",
    )
    parser.add_argument(
        "--vlm-rate-limit",
        type=float,
        default=None,
        help="Max VLM requests per second (overrides config)",
    )
    args = parser.parse_args()

    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    global _app, _job_store, _worker_pool

    try:
        config = load_config(args.config)
        config = _ensure_queue_config(config)

        log_level = args.log_level or config.logging.level
        configure_logging(level=log_level)

        if args.max_concurrent is not None:
            config.queue.max_concurrent_pipelines = args.max_concurrent
        if args.vlm_rate_limit is not None:
            config.queue.vlm_max_requests_per_second = args.vlm_rate_limit

        _app, _job_store, _worker_pool = create_queue_app(config)

        pipeline = _app.config["pipeline"]
        pipeline.start()

        _job_store.start_cleanup()
        _worker_pool.start()

        server_config = config.server
        logger.info("")
        logger.info("=" * 60)
        logger.info("GlmOcr Queue Server starting on %s:%d...", server_config.host, server_config.port)
        logger.info("  Max concurrent pipelines: %s", _worker_pool._max_concurrent)
        logger.info("  Queue max size:          %s", _worker_pool._job_queue.maxsize)
        vlm_rl = _worker_pool._vlm_limiter
        if vlm_rl:
            logger.info("  VLM rate limit:          %s req/s", vlm_rl._rate)
        logger.info("Endpoints:")
        logger.info("  POST /glmocr/parse       - Submit job (async)")
        logger.info("  GET  /glmocr/status/<id> - Job status")
        logger.info("  GET  /glmocr/result/<id> - Job result")
        logger.info("  POST /glmocr/parse-sync  - Synchronous parse")
        logger.info("  GET  /health             - Health + stats")
        logger.info("  GET  /stats              - Detailed stats")
        logger.info("=" * 60)
        logger.info("")

        _app.run(
            debug=server_config.debug,
            host=server_config.host,
            port=server_config.port,
        )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Error: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        if _worker_pool is not None:
            _worker_pool.stop(timeout=30)
        if _job_store is not None:
            _job_store.stop_cleanup()
        if _app is not None and "pipeline" in _app.config:
            _app.config["pipeline"].stop()


def build_wsgi_app(config_path: Optional[str] = None) -> Flask:
    """Build a WSGI application for gunicorn deployment.

    Usage::

        gunicorn -w 8 -k sync "glmocr.server_queue:build_wsgi_app()"

    Or with a config file::

        gunicorn -w 8 -k sync "glmocr.server_queue:build_wsgi_app('/path/to/config.yaml')"
    """
    global _app, _job_store, _worker_pool

    config = load_config(config_path)
    config = _ensure_queue_config(config)
    configure_logging(level=config.logging.level)

    _app, _job_store, _worker_pool = create_queue_app(config)

    pipeline = _app.config["pipeline"]
    pipeline.start()
    _job_store.start_cleanup()
    _worker_pool.start()

    return _app


if __name__ == "__main__":
    main()