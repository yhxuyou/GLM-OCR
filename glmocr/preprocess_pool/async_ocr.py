"""Async wrapper around the SDK's synchronous :class:`OCRClient`.

The SDK ships ``OCRClient.process()`` which is a blocking HTTP call.  To
make it "fire-and-forget" without rewriting vLLM protocol handling, we
hand each request to a :class:`ThreadPoolExecutor` and track the
in-flight futures in a dict so the caller can poll for completions.

Key design choices:

* Reuse the existing :class:`OCRClient` (its connection pool, retries,
  OpenAI / Ollama translation, and error handling all come for free).
* Reuse :meth:`PageLoader.build_request_from_image` so that prompt
  mappings, image resizing and base64 encoding stay consistent with
  the rest of the SDK.
* Hand the OCR response back through a callback (``on_region_done``)
  so the dispatcher itself never blocks.  This matches the polling
  pattern in :func:`glmocr.pipeline._workers._wait_for_any`.
"""

from __future__ import annotations

import io
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, Optional

from PIL import Image

from glmocr.dataloader import PageLoader
from glmocr.ocr_client import OCRClient
from glmocr.preprocess_pool.tasks import RegionTask
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


# Payload passed to ``on_region_done``.
# response:   raw OCR response dict (or None on error)
# status:     HTTP status code (or 500 on exception)
# error:      exception text (or None)
# skip:       True if the region was "skip" and did not need OCR
RegionDoneCallback = Callable[[RegionTask, Dict], None]


class AsyncOCRDispatcher:
    """Dispatch OCR requests asynchronously over a thread pool.

    Example::

        dispatcher = AsyncOCRDispatcher(ocr_client, page_loader, max_workers=32)
        dispatcher.start()
        for region in regions:
            dispatcher.submit(region, on_done=my_callback)
        # Poll for completions:
        while dispatcher.has_inflight():
            dispatcher.drain()
        dispatcher.close()
    """

    def __init__(
        self,
        ocr_client: OCRClient,
        page_loader: PageLoader,
        max_workers: int = 32,
        on_region_done: Optional[RegionDoneCallback] = None,
    ):
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.ocr_client = ocr_client
        self.page_loader = page_loader
        self.max_workers = max_workers
        self.on_region_done = on_region_done

        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[Future, RegionTask] = {}
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create the underlying thread pool (idempotent)."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="ocr-async",
            )
            logger.debug("AsyncOCRDispatcher started with %d workers", self.max_workers)

    def close(self) -> None:
        """Shut down the thread pool.  In-flight requests still complete."""
        self._closed = True
        if self._executor is not None:
            # wait=True so we don't drop in-flight HTTP calls.
            self._executor.shutdown(wait=True)
            self._executor = None
        logger.debug("AsyncOCRDispatcher closed")

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, region_task: RegionTask) -> None:
        """Submit a region for OCR.

        Special cases:

        * ``task_type == "abandon"``: not OCR'd, callback fired with
          ``status=204``.
        * ``task_type == "skip"``: not OCR'd, callback fired with
          ``status=200, skip=True`` so the aggregator keeps the image.
        * Anything else: a real OCR request is dispatched to the pool.
        """
        if self._closed:
            raise RuntimeError("AsyncOCRDispatcher is closed")

        # Make sure the executor exists even if the caller forgot start().
        self.start()

        if region_task.task_type == "abandon":
            self._fire(
                region_task,
                {"response": None, "status": 204, "skip": False, "error": None},
            )
            return

        if region_task.task_type == "skip":
            self._fire(
                region_task,
                {"response": None, "status": 200, "skip": True, "error": None},
            )
            return

        try:
            image = Image.open(io.BytesIO(region_task.image_bytes))
            image.load()
        except Exception as e:
            logger.warning(
                "Failed to decode region %s for OCR: %s", region_task.region_id, e
            )
            self._fire(
                region_task,
                {"response": None, "status": 500, "skip": False, "error": str(e)},
            )
            return

        try:
            req = self.page_loader.build_request_from_image(
                image, region_task.task_type
            )
        except Exception as e:
            logger.warning(
                "Failed to build OCR request for region %s: %s",
                region_task.region_id,
                e,
            )
            self._fire(
                region_task,
                {"response": None, "status": 500, "skip": False, "error": str(e)},
            )
            return

        future = self._executor.submit(self.ocr_client.process, req)
        self._futures[future] = region_task

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def drain(self, timeout: float = 0.0) -> int:
        """Reap any finished futures.  Returns the number of regions
        whose callbacks were fired on this call.

        Non-blocking by default.  Set ``timeout > 0`` to do a short
        blocking wait before reaping (mirrors the semantics of
        ``as_completed(timeout=...)``).
        """
        if not self._futures:
            return 0

        if timeout > 0:
            # Block briefly to give at least one future a chance to finish.
            # This matches the pattern in
            # ``glmocr.pipeline._workers._wait_for_any``.
            done = []
            try:
                for f in self._iter_with_timeout(self._futures, timeout):
                    done.append(f)
                    break  # one is enough; we'll loop in the caller
            except Exception:
                done = [f for f in self._futures if f.done()]
        else:
            done = [f for f in self._futures if f.done()]

        for f in done:
            region_task = self._futures.pop(f, None)
            if region_task is None:
                continue
            try:
                response, status_code = f.result()
                payload = {
                    "response": response,
                    "status": status_code,
                    "skip": False,
                    "error": None,
                }
            except Exception as e:
                logger.warning(
                    "OCR call failed for region %s: %s", region_task.region_id, e
                )
                payload = {
                    "response": None,
                    "status": 500,
                    "skip": False,
                    "error": str(e),
                }
            self._fire(region_task, payload)

        return len(done)

    @staticmethod
    def _iter_with_timeout(futures, timeout: float):
        """Yield the first future that finishes within ``timeout`` seconds."""
        from concurrent.futures import wait, FIRST_COMPLETED

        if not futures:
            return
        done, _ = wait(list(futures.keys()), timeout=timeout, return_when=FIRST_COMPLETED)
        for f in done:
            yield f

    def has_inflight(self) -> bool:
        """Return True if there are unfinished OCR requests."""
        return bool(self._futures)

    def inflight_count(self) -> int:
        return len(self._futures)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire(self, region_task: RegionTask, payload: Dict) -> None:
        if self.on_region_done is None:
            return
        try:
            self.on_region_done(region_task, payload)
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("on_region_done raised: %s", e)


__all__ = ["AsyncOCRDispatcher", "RegionDoneCallback"]
