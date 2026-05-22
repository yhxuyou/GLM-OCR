"""GLM-OCR High-Concurrency Flask Service.

Multi-process pipeline pool for high concurrency.
Each worker process maintains its own Pipeline (including the layout
detection model on GPU) and processes requests dispatched via a shared
``multiprocessing.Queue``.

Architecture::

    Flask (threaded=True)
         │  submit(req_id, images)
         ▼
    [Request Queue]  ←── multiprocessing.Queue
         │
    ┌────┼────┬────┐          Worker processes (each owns a Pipeline)
    │ W1 │ W2 │ W3 │  ...     (GPU memory: one layout model per worker)
    └────┴────┴────┘
         │
    [Response Queue]  ←── multiprocessing.Queue
         │
    [Collector Thread]  ←── buffers responses, matches by request_id
         │
    Flask responds to HTTP client
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import queue
import signal
import logging
import threading
import multiprocessing
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    from flask import Flask, request, jsonify

    _FLASK_IMPORT_ERROR = None
except ImportError as e:
    Flask = None
    request = None
    jsonify = None
    _FLASK_IMPORT_ERROR = e

from glmocr.pipeline import Pipeline
from glmocr.config import load_config
from glmocr.utils.logging import get_logger, configure_logging

if TYPE_CHECKING:
    from glmocr.config import GlmOcrConfig

logger = get_logger(__name__)

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUEST_QUEUE_MAXSIZE = 256
RESPONSE_QUEUE_MAXSIZE = 256
DEFAULT_REQUEST_TIMEOUT = 300
COLLECTOR_POLL_INTERVAL = 0.5
WORKER_POLL_INTERVAL = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_response(json_result, markdown_result):
    """Build response dict with both SDK native and MaaS-compatible fields."""
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


def _default_pool_size() -> int:
    """Return a sensible default for the number of pipeline workers."""
    cpu_count = multiprocessing.cpu_count()
    return max(1, min(2, cpu_count // 2))


# ---------------------------------------------------------------------------
# Worker Process (top-level function for multiprocessing spawn)
# ---------------------------------------------------------------------------


def _worker_process(
    worker_id: int,
    config_dict: Dict[str, Any],
    request_queue: multiprocessing.Queue,
    response_queue: multiprocessing.Queue,
) -> None:
    """Entry point for a pipeline worker process.

    Each worker:
    1. Deserialises config from the pickled dict
    2. Creates and starts its own Pipeline (layout model, OCR client, …)
    3. Loops:   pop request → pipe.process() → push response
    4. ``None`` on the request queue signals shutdown.
    """
    from glmocr.config import GlmOcrConfig

    wk_logger = logging.getLogger(f"{__name__}.worker-{worker_id}")
    wk_logger.info("Worker %d starting …", worker_id)

    config = GlmOcrConfig.model_validate(config_dict)
    pipeline: Optional[Pipeline] = None

    try:
        pipeline = Pipeline(config=config.pipeline)
        pipeline.start()
        wk_logger.info("Worker %d ready (Pipeline started).", worker_id)

        while True:
            try:
                msg = request_queue.get(timeout=WORKER_POLL_INTERVAL)
            except queue.Empty:
                if _parent_alive():
                    continue
                wk_logger.info("Parent process gone; worker %d exiting.", worker_id)
                break

            if msg is None:
                wk_logger.info("Worker %d received shutdown signal.", worker_id)
                break

            request_id = msg["request_id"]
            images: List[str] = msg["images"]
            save_layout_visualization: bool = msg.get("save_layout_visualization", False)

            wk_logger.info(
                "Worker %d processing request %s (%d image(s)) …",
                worker_id,
                request_id,
                len(images),
            )

            try:
                messages = [{"role": "user", "content": []}]
                for image_url in images:
                    messages[0]["content"].append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )

                request_data = {"messages": messages}
                results = list(
                    pipeline.process(
                        request_data,
                        save_layout_visualization=save_layout_visualization,
                    )
                )

                if not results:
                    resp_payload = _build_response(None, "")
                elif len(results) == 1:
                    r = results[0]
                    resp_payload = _build_response(
                        r.json_result, r.markdown_result or ""
                    )
                else:
                    json_result = [r.json_result for r in results]
                    markdown_result = "\n\n---\n\n".join(
                        r.markdown_result or "" for r in results
                    )
                    resp_payload = _build_response(json_result, markdown_result)

                response_queue.put(
                    {
                        "request_id": request_id,
                        "status": "success",
                        "result": resp_payload,
                    }
                )
                wk_logger.info("Worker %d completed request %s.", worker_id, request_id)

            except Exception as e:
                wk_logger.error(
                    "Worker %d error processing request %s: %s",
                    worker_id,
                    request_id,
                    e,
                )
                wk_logger.debug(traceback.format_exc())
                response_queue.put(
                    {
                        "request_id": request_id,
                        "status": "error",
                        "error": str(e),
                    }
                )

    except Exception as e:
        wk_logger.error("Worker %d failed to initialise: %s", worker_id, e)
        wk_logger.debug(traceback.format_exc())
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        wk_logger.info("Worker %d stopped.", worker_id)


def _parent_alive() -> bool:
    """Check whether the parent process is still alive."""
    try:
        return os.getppid() > 1
    except (OSError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Pipeline Pool  (runs in the main process)
# ---------------------------------------------------------------------------


class _ResponseCollector:
    """Thread-safe buffer that matches responses to waiting request threads.

    One instance lives in the main process.  A dedicated collector thread
    drains ``response_queue`` and stores results in a dict.  Request threads
    call ``wait()`` and block until their response arrives (or timeout).
    """

    def __init__(self, response_queue: multiprocessing.Queue):
        self._queue = response_queue
        self._lock = threading.Lock()
        self._results: Dict[str, Dict[str, Any]] = {}
        self._events: Dict[str, threading.Event] = {}

    def register(self, request_id: str) -> None:
        with self._lock:
            self._events[request_id] = threading.Event()

    def unregister(self, request_id: str) -> None:
        with self._lock:
            self._events.pop(request_id, None)
            self._results.pop(request_id, None)

    def wait(self, request_id: str, timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Optional[Dict[str, Any]]:
        event = self._events.get(request_id)
        if event is None:
            return None
        event.wait(timeout=timeout)
        with self._lock:
            return self._results.pop(request_id, None)

    def collect(self) -> None:
        """Drain the response queue (runs in collector thread)."""
        while True:
            try:
                resp = self._queue.get(timeout=COLLECTOR_POLL_INTERVAL)
            except queue.Empty:
                continue
            if resp is None:
                break
            request_id = resp["request_id"]
            with self._lock:
                self._results[request_id] = resp
                event = self._events.get(request_id)
                if event is not None:
                    event.set()


class PipelinePool:
    """Pool of ``Pipeline`` worker processes.

    Args:
        num_workers: Number of worker processes.
        config: Fully-resolved ``GlmOcrConfig``.
    """

    def __init__(self, num_workers: int, config: GlmOcrConfig):
        self.num_workers = num_workers
        self.config = config

        self.request_queue: multiprocessing.Queue = multiprocessing.Queue(
            maxsize=REQUEST_QUEUE_MAXSIZE
        )
        self.response_queue: multiprocessing.Queue = multiprocessing.Queue(
            maxsize=RESPONSE_QUEUE_MAXSIZE
        )
        self._collector = _ResponseCollector(self.response_queue)
        self._processes: List[multiprocessing.Process] = []
        self._collector_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start all workers and the collector thread."""
        logger.info("Starting PipelinePool with %d worker(s) …", self.num_workers)
        config_dict = self.config.model_dump()

        for i in range(self.num_workers):
            p = multiprocessing.Process(
                target=_worker_process,
                args=(i, config_dict, self.request_queue, self.response_queue),
                daemon=True,
            )
            p.start()
            self._processes.append(p)
            logger.info("Worker %d started (PID %d).", i, p.pid)

        self._collector_thread = threading.Thread(
            target=self._collector.collect,
            daemon=True,
        )
        self._collector_thread.start()
        logger.info("Collector thread started.")

    def stop(self):
        """Gracefully stop all workers and the collector."""
        logger.info("Stopping PipelinePool …")

        for _ in self._processes:
            try:
                self.request_queue.put_nowait(None)
            except (queue.Full, ValueError):
                pass

        for p in self._processes:
            if p.is_alive():
                p.join(timeout=10)
                if p.is_alive():
                    logger.warning("Terminating worker PID %d …", p.pid)
                    p.terminate()
                    p.join(timeout=5)

        self._processes.clear()

        try:
            self.response_queue.put_nowait(None)
        except (queue.Full, ValueError):
            pass

        if self._collector_thread is not None:
            self._collector_thread.join(timeout=5)
            self._collector_thread = None

        logger.info("PipelinePool stopped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        request_id: str,
        images: List[str],
        save_layout_visualization: bool = False,
    ) -> None:
        """Submit a parse request to the pool.

        Args:
            request_id: Unique identifier for this request.
            images: List of image URLs / file paths.
            save_layout_visualization: Whether to save layout visualisations.
        """
        self._collector.register(request_id)
        self.request_queue.put(
            {
                "request_id": request_id,
                "images": images,
                "save_layout_visualization": save_layout_visualization,
            }
        )

    def collect(
        self,
        request_id: str,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> Optional[Dict[str, Any]]:
        """Wait for a submitted request to complete.

        Args:
            request_id: The request ID to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            Response dict (with ``"status"`` and ``"result"`` / ``"error"``
            keys) or ``None`` on timeout.
        """
        try:
            return self._collector.wait(request_id, timeout=timeout)
        finally:
            self._collector.unregister(request_id)

    def health(self) -> Dict[str, Any]:
        """Return pool health info."""
        alive = sum(1 for p in self._processes if p.is_alive())
        return {
            "pool_size": self.num_workers,
            "active_workers": alive,
            "all_alive": alive == self.num_workers,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ---------------------------------------------------------------------------
# Flask App Factory
# ---------------------------------------------------------------------------


def create_high_concurrency_app(
    config: GlmOcrConfig,
    pool_size: Optional[int] = None,
) -> Flask:
    """Create a high-concurrency Flask app backed by a ``PipelinePool``.

    Args:
        config: Fully-resolved ``GlmOcrConfig``.
        pool_size: Number of pipeline worker processes. Auto-detected when
            ``None`` (default: ``min(2, cpu_count // 2)``).

    Returns:
        Flask application instance.
    """
    if Flask is None:
        raise ImportError(
            "Flask server support requires the optional server extra. "
            "Install with: pip install 'glmocr[server]'"
        ) from _FLASK_IMPORT_ERROR

    app = Flask(__name__)

    if pool_size is None:
        pool_size = _default_pool_size()
    pool_size = max(1, int(pool_size))

    pool = PipelinePool(num_workers=pool_size, config=config)
    pool.start()
    app.config["pool"] = pool
    app.config["doc_config"] = config

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

        request_id = f"req-{uuid.uuid4().hex[:24]}"
        save_layout_visualization = data.get("save_layout_visualization", False)
        timeout = data.get("timeout", DEFAULT_REQUEST_TIMEOUT)

        pool: PipelinePool = app.config["pool"]

        try:
            pool.submit(request_id, images, save_layout_visualization)
            response = pool.collect(request_id, timeout=float(timeout))

            if response is None:
                return jsonify({"error": "Request timed out"}), 504

            if response["status"] == "error":
                return jsonify({"error": response["error"]}), 500

            return jsonify(response["result"]), 200

        except Exception as e:
            logger.error("Parse error: %s", e)
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

    @app.route("/health", methods=["GET"])
    def health():
        pool: PipelinePool = app.config["pool"]
        return jsonify({"status": "ok", **pool.health()}), 200

    @app.route("/pool/stats", methods=["GET"])
    def pool_stats():
        pool: PipelinePool = app.config["pool"]
        return jsonify(pool.health()), 200

    return app


def main():
    """Entrypoint for the high-concurrency server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GlmOcr High-Concurrency Server"
    )
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Number of pipeline worker processes (default: auto)",
    )
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)

    app = None
    pool: Optional[PipelinePool] = None

    try:
        config = load_config(args.config)

        log_level = args.log_level or config.logging.level
        configure_logging(level=log_level)

        if args.pool_size is not None:
            pool_size = args.pool_size
        else:
            pool_size = _default_pool_size()
        pool_size = max(1, int(pool_size))

        server_config = config.server

        logger.info("")
        logger.info("=" * 60)
        logger.info("GlmOcr High-Concurrency Server")
        logger.info("Listening on %s:%d", server_config.host, server_config.port)
        logger.info("Pipeline pool size: %d", pool_size)
        logger.info("API endpoint: /glmocr/parse")
        logger.info("=" * 60)
        logger.info("")

        app = create_high_concurrency_app(config, pool_size=pool_size)
        pool = app.config["pool"]

        app.run(
            debug=server_config.debug,
            host=server_config.host,
            port=server_config.port,
            threaded=True,
        )

    except KeyboardInterrupt:
        logger.info("Shutting down …")
    except Exception as e:
        logger.error("Error: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        if pool is not None:
            pool.stop()


if __name__ == "__main__":
    main()