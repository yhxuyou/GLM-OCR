"""Preprocess pool data contracts.

This package provides:

* :class:`Region` — single cropped region produced by user preprocessing.
* :class:`DocumentPreprocessor` — protocol that user implementations satisfy.

It is intentionally framework-free: no imports from ``Pillow`` at module top
beyond the type hint, so child processes that load this file in isolation
do not pull the entire SDK into memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class Region:
    """A single recognition region output by preprocessing.

    Attributes:
        image: Cropped PIL image of the region.
        bbox: Pixel-space bbox in the original image ``(x1, y1, x2, y2)``.
            For multi-page documents the caller may encode the page index
            inside ``metadata`` (``{"page_idx": 0}`` etc.).
        task_type: One of ``"text"``, ``"table"``, ``"formula"``,
            ``"skip"`` (keep the image but do not OCR) or ``"abandon"``
            (discard entirely).
        label: Optional original label coming from the layout detector.
        polygon: Optional polygon in the same coordinate system as ``bbox``.
        metadata: Free-form user metadata forwarded to the final result.
    """

    image: "Image.Image"  # type: ignore[name-defined]  # noqa: F821
    bbox: tuple
    task_type: str
    label: str = ""
    polygon: Optional[list] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DocumentPreprocessor(Protocol):
    """User-implemented document preprocessing pipeline.

    The four stages the user is expected to run inside ``process()`` are:

    1. Document detection
    2. Document orientation detection
    3. Document distortion correction
    4. Layout detection

    Implementations are heavy (they load models), so the pool creates one
    instance **per worker process** via a factory callable.
    """

    def process(
        self,
        image: "Image.Image",  # type: ignore[name-defined]  # noqa: F821
        **kwargs: Any,
    ) -> List[Region]:
        """Run the four-stage pipeline and return detected regions.

        Args:
            image: Original PIL image (already loaded by the pool).
            **kwargs: Forwarded from :meth:`PreprocessPool.submit`.

        Returns:
            A list of :class:`Region` objects.  An empty list is valid
            (no recognisable content on the page).
        """
        ...


__all__ = ["Region", "DocumentPreprocessor"]
