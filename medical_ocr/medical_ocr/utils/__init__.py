"""Medical OCR - Utility Functions.

Common utilities for medical OCR processing.
"""

from .cache import SimpleCache
from .config import load_medical_config
from .logger import setup_logging

__all__ = [
    "SimpleCache",
    "load_medical_config",
    "setup_logging",
]
