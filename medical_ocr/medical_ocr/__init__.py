"""Medical OCR - Specialized OCR for medical documents.

A specialized OCR system built on glmocr, optimized for medical documents with:
- Multi-stage image preprocessing (YOLO, RapidOCR, UVDoc)
- Medical-specific layout detection
- Coordinate-aware result formatting
- High-performance pipeline pool for production deployment
"""

__version__ = "1.0.0"

# ============================================================================
# Core Classes (with lazy import for graceful degradation)
# ============================================================================

try:
    from .page_loader import MedicalPageLoader
except Exception as e:
    import warnings
    warnings.warn(f"Could not import MedicalPageLoader: {e}")
    MedicalPageLoader = None

try:
    from .layout_detector import MedicalLayoutDetector
except Exception as e:
    import warnings
    warnings.warn(f"Could not import MedicalLayoutDetector: {e}")
    MedicalLayoutDetector = None

try:
    from .result_formatter import MedicalResultFormatter
except Exception as e:
    import warnings
    warnings.warn(f"Could not import MedicalResultFormatter: {e}")
    MedicalResultFormatter = None

try:
    from .pipeline import MedicalOcrPipeline
except Exception as e:
    import warnings
    warnings.warn(f"Could not import MedicalOcrPipeline: {e}")
    MedicalOcrPipeline = None

try:
    from .pipeline_pool import (
        PipelinePool,
        SingleGPUPipelinePool,
        MultiGPUPipelinePool,
        PipelinePoolConfig,
        create_pipeline_pool,
    )
except Exception as e:
    import warnings
    warnings.warn(f"Could not import PipelinePool: {e}")
    PipelinePool = None
    SingleGPUPipelinePool = None
    MultiGPUPipelinePool = None
    PipelinePoolConfig = None
    create_pipeline_pool = None

try:
    from .high_perf_server import create_app, run_server
except Exception as e:
    import warnings
    warnings.warn(f"Could not import HighPerfServer: {e}")
    create_app = None
    run_server = None

# ============================================================================
# Submodules
# ============================================================================

from . import utils
from . import constants

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Core classes
    "MedicalPageLoader",
    "MedicalLayoutDetector",
    "MedicalResultFormatter",
    "MedicalOcrPipeline",
    
    # Pipeline pool
    "PipelinePool",
    "SingleGPUPipelinePool",
    "MultiGPUPipelinePool",
    "PipelinePoolConfig",
    "create_pipeline_pool",
    
    # Server
    "create_app",
    "run_server",
    
    # Submodules
    "utils",
    "constants",
]
