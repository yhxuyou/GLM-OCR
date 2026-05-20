"""Medical OCR Pipeline - Inherits from glmocr.Pipeline.

Only changes __init__ to use medical-specific subclasses, keeps all other logic identical.
"""

from typing import Optional

from glmocr.pipeline import Pipeline
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class MedicalOcrPipeline(Pipeline):
    """Medical OCR Pipeline, inherits from glmocr.Pipeline.

    All logic is identical to parent class. Only __init__ is overridden
    to use MedicalPageLoader and MedicalLayoutDetector.
    """

    def __init__(
        self,
        config,
        yolo_model_dir: Optional[str] = None,
        uvdoc_model_dir: Optional[str] = None,
        layout_detector=None,
        result_formatter=None,
    ):
        """Initialize MedicalOcrPipeline.

        Args:
            config: PipelineConfig instance.
            yolo_model_dir: Path to YOLO model for document detection
            uvdoc_model_dir: Path to UVDoc model for distortion correction
            layout_detector: Custom layout detector (optional).
            result_formatter: Custom result formatter (optional).
        """
        from .page_loader import MedicalPageLoader
        from .layout_detector import MedicalLayoutDetector
        from glmocr.ocr_client import OCRClient
        from glmocr.postprocess import ResultFormatter

        self.config = config

        # Use MedicalPageLoader instead of PageLoader
        self.page_loader = MedicalPageLoader(
            config.page_loader,
            yolo_model_dir=yolo_model_dir,
            uvdoc_model_dir=uvdoc_model_dir,
            enable_preprocessing=True,
        )

        # Standard OCRClient
        self.ocr_client = OCRClient(config.ocr_api)

        # ResultFormatter
        self.result_formatter = (
            result_formatter
            if result_formatter is not None
            else ResultFormatter(config.result_formatter)
        )

        # Use MedicalLayoutDetector or custom one
        if layout_detector is not None:
            self.layout_detector = layout_detector
        else:
            self.layout_detector = MedicalLayoutDetector(config.layout)

        # Rest of config (copied from parent)
        self.max_workers = config.max_workers
        self._page_maxsize = getattr(config, "page_maxsize", 100)
        self._region_maxsize = getattr(config, "region_maxsize", 2000)
        self._current_state = None
