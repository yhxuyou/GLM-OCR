"""User-facing entry point: 4-process preprocess pool + async OCR.

End-to-end flow::

    submit(image, document_id, on_done=cb)
        │
        ▼
    ┌──────────────────────┐
    │ multiprocessing.Queue│  (parent → 4 child processes)
    └──────────────────────┘
        │
        ▼  (each child runs doc_detect → orientation → distortion → layout)
    ┌──────────────────────┐
    │ multiprocessing.Queue│  (children → parent, region by region)
    └──────────────────────┘
        │
        ▼  (parent's reader thread)
    AsyncOCRDispatcher.submit(...)
        │
        ▼  (ThreadPoolExecutor, fires HTTP calls to local vLLM)
    AsyncOCRDispatcher.on_region_done(...)
        │
        ▼
    DocumentAggregator.on_region(...)  →  cb(result) when the doc is done

Public surface:

* :class:`PreprocessPool`  — the orchestrator.
* :meth:`PreprocessPool.submit` — push one image, get a task_id back.
* :meth:`PreprocessPool.join`  — block until everything submitted so far
  has been OCR'd and aggregated.
* Context-manager support so users can write
  ``with PreprocessPool(...) as pool: ...``.
"""

from __future__ import annotations

import io
import os
import queue as _queue
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue as MpQueue
from typing import Any, Callable, Dict, List, Optional, Union

from PIL import Image

from glmocr.config import GlmOcrConfig
from glmocr.dataloader import PageLoader
from glmocr.ocr_client import OCRClient
from glmocr.preprocess_pool.aggregator import DocumentAggregator, DocumentResult
from glmocr.preprocess_pool.async_ocr import AsyncOCRDispatcher
from glmocr.preprocess_pool.preprocess_worker import preprocess_worker
from glmocr.preprocess_pool.protocols import DocumentPreprocessor
from glmocr.preprocess_pool.tasks import (
    PreprocessTask,
    RegionError,
    RegionTask,
    WORKER_STOP_SENTINEL,
)
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


# Type alias for the per-document result delivered to user callbacks.
OnDoneCallback = Callable[[DocumentResult], None]


@dataclass
class _SubmitContext:
    """Metadata kept on the parent side for each submitted task."""

    document_id: str
    task_id: str
    expected_regions: Optional[int]
    on_done: Optional[OnDoneCallback]
    source_image_path: Optional[str]


# Sentinel returned when a submit call gets an image in an unsupported form.
class _PoolClosedError(RuntimeError):
    pass


class PreprocessPool:
    """4-process preprocessing pool with async OCR and per-doc aggregation.

    Args:
        preprocessor: Zero-arg factory that returns a fresh
            :class:`DocumentPreprocessor`.  Called once per worker process
            (so model loads happen in parallel, one per pid).
        config: A fully-built :class:`GlmOcrConfig`.  Only the
            ``ocr_api`` and ``page_loader`` sub-sections are used here.
        num_workers: Number of child processes (default 4).
        ocr_max_workers: Size of the OCR :class:`ThreadPoolExecutor`
            (default 32).
        ocr_client: Optional pre-built :class:`OCRClient` (else one is
            built from ``config``).
        page_loader: Optional pre-built :class:`PageLoader`.
        reader_poll_interval: Seconds between reader-thread polls.
        result_queue_maxsize: Bound on the cross-process output queue.
        input_queue_maxsize: Bound on the cross-process input queue.

    Example::

        class MyDocPreprocessor:
            def process(self, image, **_):
                # 1) doc detection, 2) orientation, 3) distortion fix, 4) layout
                return [Region(image=crop, bbox=(x1,y1,x2,y2), task_type="text")]

        cfg = GlmOcrConfig.from_env()
        with PreprocessPool(MyDocPreprocessor, cfg) as pool:
            pool.submit(image_path="a.png", document_id="doc1", on_done=cb)
            pool.submit(image_path="b.png", document_id="doc1", on_done=cb,
                        expected_regions=2)
            pool.join()
    """

    def __init__(
        self,
        preprocessor: Callable[[], DocumentPreprocessor],
        config: GlmOcrConfig,
        num_workers: int = 4,
        ocr_max_workers: int = 32,
        ocr_client: Optional[OCRClient] = None,
        page_loader: Optional[PageLoader] = None,
        reader_poll_interval: float = 0.1,
        result_queue_maxsize: int = 2000,
        input_queue_maxsize: int = 200,
    ):
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if ocr_max_workers <= 0:
            raise ValueError("ocr_max_workers must be positive")

        self._preprocessor_factory = preprocessor
        self._config = config
        self._num_workers = num_workers
        self._ocr_max_workers = ocr_max_workers
        self._reader_poll_interval = reader_poll_interval
        self._result_queue_maxsize = result_queue_maxsize
        self._input_queue_maxsize = input_queue_maxsize

        self._ocr_client = ocr_client or OCRClient(config.pipeline.ocr_api)
        self._page_loader = page_loader or PageLoader(config.pipeline.page_loader)

        # Multiprocessing primitives
        self._input_q: Optional[MpQueue] = None
        self._output_q: Optional[MpQueue] = None
        self._processes: List[Process] = []

        # Main-process primitives
        self._aggregator = DocumentAggregator()
        self._dispatcher: Optional[AsyncOCRDispatcher] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_exception: Optional[BaseException] = None
        self._stop_event = threading.Event()

        # Per-submit metadata (main process only).
        self._sub_lock = threading.Lock()
        self._submits: Dict[str, _SubmitContext] = {}
        self._started = False
        self._closed = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "PreprocessPool":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker processes, OCR dispatcher, and reader thread."""
        if self._started:
            raise RuntimeError("PreprocessPool already started")
        if self._closed:
            raise RuntimeError("PreprocessPool is closed; create a new instance")

        # Connect to the OCR backend first so that misconfiguration is
        # surfaced immediately, not at first submit.
        self._ocr_client.start()

        self._input_q = MpQueue(maxsize=self._input_queue_maxsize)
        self._output_q = MpQueue(maxsize=self._result_queue_maxsize)

        for i in range(self._num_workers):
            p = Process(
                target=preprocess_worker,
                args=(
                    self._input_q,
                    self._output_q,
                    self._preprocessor_factory,
                ),
                name=f"glmocr-preprocess-{i}",
                daemon=True,
            )
            p.start()
            self._processes.append(p)
        logger.info(
            "PreprocessPool started with %d workers (pids=%s)",
            self._num_workers,
            [p.pid for p in self._processes],
        )

        self._dispatcher = AsyncOCRDispatcher(
            ocr_client=self._ocr_client,
            page_loader=self._page_loader,
            max_workers=self._ocr_max_workers,
            on_region_done=self._on_region_done,
        )
        self._dispatcher.start()

        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._result_reader_loop, name="glmocr-pool-reader", daemon=True
        )
        self._reader_thread.start()
        self._started = True

    def stop(self, timeout: float = 10.0) -> None:
        """Stop everything in order: workers → reader → dispatcher → OCR."""
        if not self._started:
            return
        if self._closed:
            return
        self._closed = True

        # 1. Stop child processes.
        if self._input_q is not None:
            for _ in self._processes:
                try:
                    self._input_q.put(WORKER_STOP_SENTINEL)
                except Exception:  # pragma: no cover - queue may be broken
                    pass
            for p in self._processes:
                p.join(timeout=timeout)
                if p.is_alive():
                    logger.warning("Process %s did not exit in time, terminating", p.name)
                    p.terminate()
                    p.join(timeout=2.0)
            self._processes.clear()

        # 2. Stop the reader thread.
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)
            if self._reader_thread.is_alive():
                logger.warning("Reader thread did not exit cleanly")
            self._reader_thread = None

        # 3. Drain & close the OCR dispatcher.
        if self._dispatcher is not None:
            try:
                self._dispatcher.drain(timeout=2.0)
            except Exception:  # pragma: no cover
                pass
            self._dispatcher.close()
            self._dispatcher = None

        # 4. Close the OCR client.
        try:
            self._ocr_client.stop()
        except Exception:  # pragma: no cover
            logger.exception("Error stopping OCR client")

        logger.info("PreprocessPool stopped")

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        pil_image: Optional[Image.Image] = None,
        document_id: str = "",
        on_done: Optional[OnDoneCallback] = None,
        expected_regions: Optional[int] = None,
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Submit a document image to the pool.

        Provide the image in exactly one of three forms: ``image_path``,
        ``image_bytes`` or ``pil_image``.

        Args:
            image_path: Local path to an image file.
            image_bytes: Raw encoded image bytes (any PIL-supported format).
            pil_image: An already-decoded :class:`PIL.Image.Image`.
            document_id: Logical id used by the aggregator.  All regions
                coming from this submit (and any future submits with the
                same id) will be merged into the same :class:`DocumentResult`.
            on_done: Callback fired once the document is finalised.
            expected_regions: Total regions the user expects for this doc.
                If set, the aggregator will fire ``on_done`` automatically
                when that many region results have arrived.  ``None``
                means the user will call :meth:`flush_document` themselves.
            preprocess_kwargs: Forwarded to ``DocumentPreprocessor.process``.
            task_id: Optional explicit id; auto-generated if omitted.

        Returns:
            The ``task_id`` assigned to this submission.
        """
        if not self._started:
            raise _PoolClosedError("PreprocessPool is not started")
        if self._closed:
            raise _PoolClosedError("PreprocessPool is closed")
        if not document_id:
            raise ValueError("document_id is required")

        # --- Decode the image ---
        if image_path is not None:
            with open(image_path, "rb") as f:
                data = f.read()
        elif image_bytes is not None:
            data = image_bytes
        elif pil_image is not None:
            buf = io.BytesIO()
            if pil_image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
                pil_image = pil_image.convert("RGB")
            pil_image.save(buf, format="PNG")
            data = buf.getvalue()
        else:
            raise ValueError(
                "submit() requires exactly one of image_path, image_bytes, or pil_image"
            )

        # --- Track metadata on the parent side ---
        if task_id is None:
            task_id = f"t-{int(time.time() * 1e6)}-{os.getpid()}-{id(self)}"
        with self._sub_lock:
            self._submits[task_id] = _SubmitContext(
                document_id=document_id,
                task_id=task_id,
                expected_regions=expected_regions,
                on_done=on_done,
                source_image_path=image_path,
            )

        # --- Register the document with the aggregator (idempotent). ---
        if on_done is not None or expected_regions is not None:
            self._aggregator.register(
                document_id=document_id,
                on_done=on_done,
                expected_regions=expected_regions,
                source_image_path=image_path,
            )

        # --- Push the task to the worker pool. ---
        assert self._input_q is not None
        self._input_q.put(
            PreprocessTask(
                task_id=task_id,
                document_id=document_id,
                image_bytes=data,
                on_done=None,
                preprocess_kwargs=preprocess_kwargs or {},
            )
        )
        return task_id

    def flush_document(self, document_id: str) -> Optional[DocumentResult]:
        """Force-finalise a document.  Use when ``expected_regions=None``."""
        return self._aggregator.flush_document(document_id)

    # ------------------------------------------------------------------
    # Blocking helpers
    # ------------------------------------------------------------------

    def join(self, timeout: Optional[float] = None) -> bool:
        """Block until all submitted work has been finalised.

        Returns:
            ``True`` if everything completed, ``False`` on timeout.
        """
        deadline = (time.time() + timeout) if timeout is not None else None
        while True:
            inflight = self._inflight_count()
            if inflight == 0 and not self._aggregator.has_pending():
                return True
            if deadline is not None and time.time() > deadline:
                logger.warning(
                    "PreprocessPool.join() timed out; %d items still in flight",
                    inflight,
                )
                return False
            time.sleep(self._reader_poll_interval)

    def _inflight_count(self) -> int:
        count = 0
        if self._input_q is not None:
            try:
                count += self._input_q.qsize()
            except Exception:  # pragma: no cover
                pass
        if self._output_q is not None:
            try:
                count += self._output_q.qsize()
            except Exception:  # pragma: no cover
                pass
        if self._dispatcher is not None:
            count += self._dispatcher.inflight_count()
        return count

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------

    def _result_reader_loop(self) -> None:
        """Pull region/error messages from the child processes and dispatch."""
        assert self._output_q is not None
        assert self._dispatcher is not None
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._output_q.get(timeout=self._reader_poll_interval)
                except _queue.Empty:
                    # No new regions; still drain the dispatcher.
                    self._dispatcher.drain()
                    continue
                except (EOFError, OSError):  # pragma: no cover
                    break

                if isinstance(item, RegionError):
                    self._handle_region_error(item)
                elif isinstance(item, RegionTask):
                    self._dispatcher.submit(item)
                else:
                    logger.warning("Unknown message from child: %r", item)

                # Reap finished OCR requests on every iteration.
                self._dispatcher.drain()
        except BaseException as e:  # pragma: no cover - defensive
            self._reader_exception = e
            logger.exception("Result reader crashed: %s", e)
            logger.debug("Traceback: %s", traceback.format_exc())
            self._stop_event.set()

    def _handle_region_error(self, item: RegionError) -> None:
        """Convert a whole-task preprocessing failure into a region result."""
        logger.error(
            "Preprocessing failed for document %s task %s: %s",
            item.document_id,
            item.task_id,
            item.error,
        )
        # Synthesise a single "abandon" region so the aggregator can
        # advance its counter if the user declared expected_regions.
        synthetic = RegionTask(
            region_id=f"{item.task_id}#err",
            document_id=item.document_id,
            task_id=item.task_id,
            image_bytes=b"",
            bbox=(0, 0, 0, 0),
            task_type="abandon",
            label="",
            polygon=None,
            on_done=None,
            metadata={"error": item.error},
        )
        self._dispatcher.submit(synthetic)
        # Make sure the callback fires promptly for the failure case.
        self._dispatcher.drain()

    def _on_region_done(
        self, region_task: RegionTask, payload: Dict[str, Any]
    ) -> None:
        """AsyncOCRDispatcher callback: aggregate the result."""
        self._aggregator.on_region(region_task, payload)


__all__ = ["PreprocessPool"]
