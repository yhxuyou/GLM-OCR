"""Medical OCR - Configuration Utilities.

Configuration loading and merging.
"""

import os
from typing import Optional, Any

from glmocr.config import load_config


def load_medical_config(config_path: Optional[str] = None) -> Any:
    """Load Medical OCR configuration.

    Args:
        config_path: Optional path to config file

    Returns:
        Configuration object
    """
    if config_path and os.path.exists(config_path):
        return load_config(config_path)

    default_paths = [
        "./config.yaml",
        "./config_simple.yaml",
        "../config.yaml",
        "/workspace/medical_ocr/config.yaml",
    ]

    for path in default_paths:
        if os.path.exists(path):
            return load_config(path)

    return load_config()


def merge_config(base_config: Any, overrides: dict) -> Any:
    """Merge override values into base config.

    Args:
        base_config: Base configuration object
        overrides: Dictionary of override values

    Returns:
        Merged configuration
    """
    import copy

    config = copy.deepcopy(base_config)

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
