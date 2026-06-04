"""Preprocess pool: 4-process image preprocessing + async OCR + per-document aggregation.

Public surface::

    from glmocr.preprocess_pool import (
        # Pool + base contracts
        PreprocessPool, DocumentPreprocessor, Region,
        # 4-stage detector base classes
        BaseDocumentDetector, BaseOrientationDetector,
        BaseDistortionCorrector, LayoutDetectorAdapter,
        DocumentBox, Orientation,
        # Orchestrator
        DocumentPreprocessorPipeline,
        # Internals (rarely needed)
        DocumentAggregator, DocumentResult, AsyncOCRDispatcher,
        RegionDoneCallback, PreprocessTask, RegionTask, RegionError,
        WORKER_STOP_SENTINEL,
    )
"""

from .aggregator import DocumentAggregator, DocumentResult
from .async_ocr import AsyncOCRDispatcher, RegionDoneCallback
from .detectors import (
    BaseDetector,
    BaseDistortionCorrector,
    BaseDocumentDetector,
    BaseOrientationDetector,
    DocumentBox,
    LayoutDetectorAdapter,
    Orientation,
)
from .pipeline import DocumentPreprocessorPipeline
from .pool import PreprocessPool
from .protocols import DocumentPreprocessor, Region
from .tasks import (
    WORKER_STOP_SENTINEL,
    PreprocessTask,
    RegionError,
    RegionTask,
)

__all__ = [
    # Pool + base
    "PreprocessPool",
    "DocumentPreprocessor",
    "Region",
    # Detectors
    "BaseDetector",
    "BaseDocumentDetector",
    "BaseOrientationDetector",
    "BaseDistortionCorrector",
    "LayoutDetectorAdapter",
    "DocumentBox",
    "Orientation",
    # Orchestrator
    "DocumentPreprocessorPipeline",
    # Internals
    "DocumentAggregator",
    "DocumentResult",
    "AsyncOCRDispatcher",
    "RegionDoneCallback",
    "PreprocessTask",
    "RegionTask",
    "RegionError",
    "WORKER_STOP_SENTINEL",
]
