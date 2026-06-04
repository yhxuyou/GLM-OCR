"""Cross-process message types for the preprocess pool.

All dataclasses are ``pickle``-friendly (no closures, no PIL Image objects).
Images are passed as PNG bytes to keep multiprocessing queues stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------

# Worker processes break out of their main loop when they receive this value
# on the input queue.  Defined as a module-level singleton so the parent and
# child processes see the same object identity (the value itself is ``None``
# which is naturally picklable).
WORKER_STOP_SENTINEL: Optional["PreprocessTask"] = None


# ---------------------------------------------------------------------------
# Parent → child (input queue)
# ---------------------------------------------------------------------------


@dataclass
class PreprocessTask:
    """One document image to be preprocessed.

    Attributes:
        task_id: Pool-wide unique identifier.
        document_id: Logical document id used by the aggregator to group
            regions of the same document together.
        image_bytes: Original image encoded as PNG bytes.
        on_done: Reserved for future use; child processes must ignore it
            (callbacks cannot be sent through multiprocessing queues).
        preprocess_kwargs: User-supplied kwargs forwarded to
            :meth:`DocumentPreprocessor.process`.
    """

    task_id: str
    document_id: str
    image_bytes: bytes
    on_done: Optional[Callable] = None
    preprocess_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Child → parent (output queue)
# ---------------------------------------------------------------------------


@dataclass
class RegionTask:
    """A single region produced by the preprocessor, queued for OCR.

    Attributes:
        region_id: ``f"{task_id}#{idx}"`` — unique within a task.
        document_id: Copied from the parent task.
        task_id: Source preprocess task id.
        image_bytes: Cropped region encoded as PNG.
        bbox: ``(x1, y1, x2, y2)`` in original-image pixel space.
        task_type: ``"text"`` / ``"table"`` / ``"formula"`` / ``"skip"`` /
            ``"abandon"``.
        label: Optional layout label.
        polygon: Optional polygon.
        on_done: Always ``None`` on the wire; the parent attaches the
            real callback from its own context.
        metadata: Free-form metadata from the preprocessor.
    """

    region_id: str
    document_id: str
    task_id: str
    image_bytes: bytes
    bbox: Tuple[int, int, int, int]
    task_type: str
    label: str = ""
    polygon: Optional[list] = None
    on_done: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegionError:
    """Whole-task preprocessing failure (rare; usually a crash inside the
    user's preprocessor)."""

    region_id: str
    document_id: str
    task_id: str
    error: str


__all__ = [
    "PreprocessTask",
    "RegionTask",
    "RegionError",
    "WORKER_STOP_SENTINEL",
]
