"""Detectors for the four-stage preprocessing pipeline.

Public surface (re-exported by :mod:`glmocr.preprocess_pool`):

* :class:`BaseDetector`           — common lifecycle / abstract hooks
* :class:`BaseDocumentDetector`   — Stage 1 skeleton
* :class:`BaseOrientationDetector`— Stage 2 skeleton
* :class:`BaseDistortionCorrector`— Stage 3 skeleton
* :class:`LayoutDetectorAdapter`  — Stage 4 adapter (fully working)
* :class:`DocumentBox`            — Stage 1 output dataclass
* :class:`Orientation`            — Stage 2 output enum
"""

from .base import BaseDetector, DocumentBox, Orientation
from .distortion_corrector import BaseDistortionCorrector
from .document_detector import BaseDocumentDetector
from .layout_adapter import LayoutDetectorAdapter
from .orientation_detector import BaseOrientationDetector

__all__ = [
    "BaseDetector",
    "BaseDocumentDetector",
    "BaseOrientationDetector",
    "BaseDistortionCorrector",
    "LayoutDetectorAdapter",
    "DocumentBox",
    "Orientation",
]
