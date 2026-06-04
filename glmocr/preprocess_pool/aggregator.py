"""Per-document result aggregation.

The :class:`DocumentAggregator` is the single point in the pool where
regions belonging to the same logical document are stitched together.
The aggregator is also responsible for firing the user-supplied
``on_done`` callback exactly once per document.

The completion policy is intentionally simple:

* If the user calls :meth:`register` with ``expected_regions=None`` the
  aggregator buffers regions forever and never finalises — the user is
  expected to call :meth:`flush_document` when their application logic
  decides the document is complete (e.g. when a separate page tracker
  reports "all pages sent and OCR'd").
* If ``expected_regions`` is an integer, the aggregator finalises the
  document as soon as that many region results have arrived.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from glmocr.preprocess_pool.tasks import RegionTask
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class _PendingDoc:
    document_id: str
    expected: int
    received: int = 0
    regions: List[Dict[str, Any]] = field(default_factory=list)
    on_done: Optional[Callable[[Dict[str, Any]], None]] = None
    source_image_path: Optional[str] = None


# Result dict shape delivered to the user callback.
DocumentResult = Dict[str, Any]


class DocumentAggregator:
    """Group region results by document and emit a single callback per doc.

    Thread safety: all mutating methods use an internal lock, so it is
    safe to call :meth:`on_region` from the OCR dispatcher thread while
    :meth:`register` / :meth:`flush_document` are called from the main
    thread.
    """

    def __init__(self):
        import threading

        self._pending: Dict[str, _PendingDoc] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        document_id: str,
        on_done: Optional[Callable[[DocumentResult], None]] = None,
        expected_regions: Optional[int] = None,
        source_image_path: Optional[str] = None,
    ) -> None:
        """Begin tracking a new document.

        Args:
            document_id: Logical document id used by :class:`PreprocessPool`.
            on_done: Callback fired once the document is finalised.
            expected_regions: Total number of regions the user expects.
                ``None`` means "I'll call :meth:`flush_document` myself".
            source_image_path: Original image path stored in the result.
        """
        with self._lock:
            if document_id in self._pending:
                # Update the callback / expected in case of re-submission.
                existing = self._pending[document_id]
                if on_done is not None:
                    existing.on_done = on_done
                if expected_regions is not None:
                    existing.expected = expected_regions
                if source_image_path is not None:
                    existing.source_image_path = source_image_path
                return

            self._pending[document_id] = _PendingDoc(
                document_id=document_id,
                expected=expected_regions or 0,
                on_done=on_done,
                source_image_path=source_image_path,
            )

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------

    def on_region(
        self, region_task: RegionTask, payload: Dict[str, Any]
    ) -> Optional[DocumentResult]:
        """Record a region result.

        Returns:
            The finalised document result if the document is now
            complete, otherwise ``None``.
        """
        with self._lock:
            doc = self._pending.get(region_task.document_id)
            if doc is None:
                logger.debug(
                    "Received region for unknown document %s; ignoring",
                    region_task.document_id,
                )
                return None

            # Decode skip image bytes for the final result (if any).
            skip_image = None
            if payload.get("skip") and region_task.image_bytes:
                try:
                    skip_image = Image.open(io.BytesIO(region_task.image_bytes))
                    skip_image.load()
                except Exception as e:
                    logger.warning(
                        "Failed to decode skip image for %s: %s",
                        region_task.region_id,
                        e,
                    )

            response = payload.get("response")
            content: Optional[str] = None
            if response is not None and payload.get("status") == 200:
                # Try OpenAI format first.
                try:
                    content = response["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    content = None
                if content:
                    content = content.strip()

            doc.regions.append(
                {
                    "region_id": region_task.region_id,
                    "task_id": region_task.task_id,
                    "bbox": region_task.bbox,
                    "task_type": region_task.task_type,
                    "label": region_task.label,
                    "polygon": region_task.polygon,
                    "metadata": region_task.metadata,
                    "content": content,
                    "status": payload.get("status"),
                    "skip": payload.get("skip", False),
                    "image": skip_image,  # PIL image or None
                }
            )
            doc.received += 1

            if doc.expected and doc.received >= doc.expected:
                return self._finalize_locked(doc)

        return None

    def flush_document(self, document_id: str) -> Optional[DocumentResult]:
        """Force-finalise a document (used when ``expected_regions=None``)."""
        with self._lock:
            doc = self._pending.get(document_id)
            if doc is None:
                return None
            return self._finalize_locked(doc)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def pending_document_ids(self) -> List[str]:
        with self._lock:
            return list(self._pending.keys())

    def pending_region_count(self, document_id: str) -> int:
        with self._lock:
            doc = self._pending.get(document_id)
            return doc.received if doc else 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _finalize_locked(self, doc: _PendingDoc) -> DocumentResult:
        result: DocumentResult = {
            "document_id": doc.document_id,
            "source_image_path": doc.source_image_path,
            "regions": doc.regions,
            "num_regions": len(doc.regions),
        }
        # Sort by region_id to keep the result stable (region_id encodes
        # the order in which the preprocessor produced them).
        result["regions"].sort(key=lambda r: r["region_id"])

        if doc.on_done is not None:
            try:
                doc.on_done(result)
            except Exception as e:  # pragma: no cover - defensive
                logger.exception("on_done callback raised for %s: %s", doc.document_id, e)

        self._pending.pop(doc.document_id, None)
        return result


__all__ = ["DocumentAggregator", "DocumentResult"]
