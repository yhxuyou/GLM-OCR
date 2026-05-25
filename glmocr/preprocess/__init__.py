"""Image preprocessing module.

Provides GPU-accelerated document preprocessing:
  - :class:`DocumentDetector` — detect and crop document region
  - :class:`OrientationCorrector` — rotate image to upright
  - :class:`Dewarper` — flatten curved/warped document surfaces
  - :class:`PreprocessStage` — orchestrator that runs all three in sequence

All models are small (total < 100 MB VRAM) and are loaded once at startup,
staying resident on GPU until shutdown.  See the ``base.py`` module for
the abstract interface.
"""

from glmocr.preprocess.base import BasePreprocessor
from glmocr.preprocess.document_detector import DocumentDetector
from glmocr.preprocess.orientation_corrector import OrientationCorrector
from glmocr.preprocess.dewarper import Dewarper
from glmocr.preprocess.preprocess_stage import PreprocessStage

__all__ = [
    "BasePreprocessor",
    "DocumentDetector",
    "OrientationCorrector",
    "Dewarper",
    "PreprocessStage",
]