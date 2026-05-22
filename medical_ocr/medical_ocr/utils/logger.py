"""Medical OCR - Logging Utilities.

Logging configuration and utilities.
"""

import os
import sys
import logging
from typing import Optional

from glmocr.utils.logging import get_logger


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None
) -> None:
    """Setup logging for medical OCR.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "[%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        stream=sys.stdout
    )


def get_medical_logger(name: str) -> logging.Logger:
    """Get a logger for medical OCR components.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return get_logger(name)
