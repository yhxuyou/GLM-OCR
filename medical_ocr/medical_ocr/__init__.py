"""Medical OCR - Specialized OCR for medical documents."""

__version__ = "0.1.0"

# 延迟导入以避免依赖问题
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

__all__ = ["MedicalOcrPipeline", "MedicalLayoutDetector"]
