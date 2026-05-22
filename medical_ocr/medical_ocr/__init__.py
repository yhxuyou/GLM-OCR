"""Medical OCR - Specialized OCR for medical documents."""

__version__ = "0.1.0"

# 延迟导入以避免依赖问题
try:
    from .page_loader import MedicalPageLoader
except Exception as e:
    import warnings
    warnings.warn(f"Could not import MedicalPageLoader: {e}")
    MedicalPageLoader = None

try:
    from .pipeline import MedicalOcrPipeline
except Exception as e:
    import warnings
    warnings.warn(f"Could not import MedicalOcrPipeline: {e}")
    MedicalOcrPipeline = None

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

__all__ = [
    "MedicalPageLoader", 
    "MedicalOcrPipeline", 
    "MedicalLayoutDetector",
    "MedicalResultFormatter"
]
