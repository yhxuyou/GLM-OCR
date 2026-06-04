"""Preprocess pool: 4-process image preprocessing + async OCR + per-document aggregation.

Public surface::

    from glmocr.preprocess_pool import (
        PreprocessPool,
        DocumentPreprocessor,
        Region,
    )

Users implement a :class:`DocumentPreprocessor` that runs the four
preprocessing stages (document detection, orientation detection,
distortion correction, layout detection) and returns a list of
:class:`Region` objects.  :class:`PreprocessPool` then takes care of
fanning those regions out to a thread-pool-backed ``OCRClient`` and
emitting a per-document callback when all regions of the same
``document_id`` have been processed.
"""

from .aggregator import DocumentAggregator, DocumentResult
from .async_ocr import AsyncOCRDispatcher, RegionDoneCallback
from .pool import PreprocessPool
from .protocols import DocumentPreprocessor, Region
from .tasks import (
    PreprocessTask,
    RegionError,
    RegionTask,
    WORKER_STOP_SENTINEL,
)

__all__ = [
    "PreprocessPool",
    "DocumentPreprocessor",
    "Region",
    "DocumentAggregator",
    "DocumentResult",
    "AsyncOCRDispatcher",
    "RegionDoneCallback",
    "PreprocessTask",
    "RegionTask",
    "RegionError",
    "WORKER_STOP_SENTINEL",
]
