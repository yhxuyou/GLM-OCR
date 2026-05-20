"""Medical Page Loader - Enhanced PageLoader with document preprocessing.

Features:
- Document detection using YOLO model
- Document orientation correction using RapidOCR
- Document distortion correction using UVDoc

Inherits from glmocr's PageLoader and adds preprocessing capabilities.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import cv2
import numpy as np

from glmocr.dataloader import PageLoader
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class MedicalPageLoader(PageLoader):
    """Medical document page loader with preprocessing.

    Adds three-stage preprocessing:
    1. YOLO document detection
    2. RapidOCR orientation correction
    3. UVDoc distortion correction

    Inherits all loading capabilities from PageLoader.
    """

    # Medical document specific region labels to prioritize
    MEDICAL_LABELS = {
        "table", "chart", "diagram", "signature",
        "stamp", "header", "footer", "medical_record"
    }

    def __init__(
        self,
        config,
        yolo_model_dir: Optional[str] = None,
        uvdoc_model_dir: Optional[str] = None,
        enable_preprocessing: bool = True,
    ):
        """Initialize MedicalPageLoader.

        Args:
            config: PageLoaderConfig instance
            yolo_model_dir: Path to YOLO model for document detection
            uvdoc_model_dir: Path to UVDoc model for distortion correction
            enable_preprocessing: Whether to enable preprocessing pipeline
        """
        super().__init__(config)

        self.enable_preprocessing = enable_preprocessing
        self.yolo_model_dir = yolo_model_dir
        self.uvdoc_model_dir = uvdoc_model_dir

        # Model instances (lazy initialization)
        self._yolo_model = None
        self._uvdoc_model = None
        self._rapidocr_detector = None

        # Configuration
        self.yolo_confidence_threshold = 0.5
        self.yolo_nms_threshold = 0.45
        self.document_padding = 10  # pixels
        self._models_initialized = False

    # =========================================================================
    # Model Initialization
    # =========================================================================

    def _init_models(self):
        """Initialize all models (lazy initialization)."""
        if self._models_initialized:
            return

        self._init_yolo_model()
        self._init_rapidocr()
        self._init_uvdoc_model()
        self._models_initialized = True
        logger.info("MedicalPageLoader models initialized successfully")

    def _init_yolo_model(self):
        """Initialize YOLO document detection model."""
        if not self.yolo_model_dir:
            logger.info("YOLO model directory not provided, skipping")
            return

        try:
            # Try to import ultralytics YOLO
            from ultralytics import YOLO
            self._yolo_model = YOLO(f"{self.yolo_model_dir}/best.pt")
            logger.info(f"YOLO model loaded from {self.yolo_model_dir}")
        except ImportError:
            logger.warning("ultralytics not installed, YOLO detection disabled")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._yolo_model = None

    def _init_rapidocr(self):
        """Initialize RapidOCR for orientation detection."""
        try:
            from rapidocr_onnxruntime import RapidOCRDet
            self._rapidocr_detector = RapidOCRDet()
            logger.info("RapidOCR detector initialized")
        except ImportError:
            logger.warning("rapidocr_onnxruntime not installed, orientation detection disabled")
            self._rapidocr_detector = None
        except Exception as e:
            logger.error(f"Failed to initialize RapidOCR: {e}")
            self._rapidocr_detector = None

    def _init_uvdoc_model(self):
        """Initialize UVDoc distortion correction model."""
        if not self.uvdoc_model_dir:
            logger.info("UVDoc model directory not provided, skipping")
            return

        try:
            # Try to import UVDoc
            # Create simple wrapper if UVDoc not available
            from .uvdoc_inference import UVDocInference
            self._uvdoc_model = UVDocInference(self.uvdoc_model_dir)
            logger.info(f"UVDoc model loaded from {self.uvdoc_model_dir}")
        except Exception as e:
            logger.error(f"Failed to load UVDoc model: {e}")
            self._uvdoc_model = None

    # =========================================================================
    # Document Detection (YOLO)
    # =========================================================================

    def detect_document(self, image: Image.Image) -> Optional[List[List[int]]]:
        """Detect document region using YOLO model.

        Args:
            image: PIL Image

        Returns:
            List of bounding boxes [[x1, y1, x2, y2], ...] or None if no detection
        """
        self._init_models()

        if self._yolo_model is None:
            return None

        try:
            # Convert PIL to numpy array
            img_np = np.array(image.convert("RGB"))

            # Run YOLO inference
            results = self._yolo_model(
                img_np,
                conf=self.yolo_confidence_threshold,
                iou=self.yolo_nms_threshold,
                verbose=False
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    # Return the largest detected document
                    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                    best_idx = areas.index(max(areas))
                    return boxes[best_idx].astype(int).tolist()

            return None

        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
            return None

    def crop_document(self, image: Image.Image, bbox: List[int]) -> Image.Image:
        """Crop document region with padding.

        Args:
            image: PIL Image
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Cropped PIL Image
        """
        img_np = np.array(image)
        h, w = img_np.shape[:2]

        x1, y1, x2, y2 = bbox

        # Add padding
        x1 = max(0, x1 - self.document_padding)
        y1 = max(0, y1 - self.document_padding)
        x2 = min(w, x2 + self.document_padding)
        y2 = min(h, y2 + self.document_padding)

        cropped = img_np[y1:y2, x1:x2]
        return Image.fromarray(cropped)

    # =========================================================================
    # Orientation Correction (RapidOCR)
    # =========================================================================

    def correct_orientation(self, image: Image.Image) -> Image.Image:
        """Detect and correct document orientation using RapidOCR.

        Args:
            image: PIL Image

        Returns:
            Orientation-corrected PIL Image
        """
        self._init_models()

        if self._rapidocr_detector is None:
            return image

        try:
            # Convert PIL to numpy
            img_np = np.array(image.convert("RGB"))
            h, w = img_np.shape[:2]

            # Run RapidOCR detection
            results, _, _ = self._rapidocr_detector(img_np)

            if results is None or len(results) == 0:
                return image

            # Calculate average angle from detected text boxes
            angles = []
            for box in results:
                # box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Calculate angle from first two points
                p1, p2 = box[0], box[1]
                angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
                angles.append(angle)

            avg_angle = np.median(angles)

            # Rotate if angle is significant (> 5 degrees)
            if abs(avg_angle) > 5:
                # Determine rotation direction
                if avg_angle > 0:
                    if avg_angle > 45:
                        # Rotate 90 degrees counter-clockwise
                        rotation_angle = 90
                    else:
                        rotation_angle = -avg_angle
                else:
                    if avg_angle < -45:
                        # Rotate 90 degrees clockwise
                        rotation_angle = -90
                    else:
                        rotation_angle = -avg_angle

                # Rotate image
                center = (w / 2, h / 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                rotated = cv2.warpAffine(
                    img_np,
                    rotation_matrix,
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)
                )

                logger.info(f"Rotated image by {rotation_angle:.1f} degrees")
                return Image.fromarray(rotated)

            return image

        except Exception as e:
            logger.warning(f"Orientation correction failed: {e}")
            return image

    # =========================================================================
    # Distortion Correction (UVDoc)
    # =========================================================================

    def correct_distortion(self, image: Image.Image) -> Image.Image:
        """Correct document distortion using UVDoc model.

        Args:
            image: PIL Image

        Returns:
            Distortion-corrected PIL Image
        """
        self._init_models()

        if self._uvdoc_model is None:
            return image

        try:
            img_np = np.array(image.convert("RGB"))

            # UVDoc inference
            corrected_np = self._uvdoc_model.process(img_np)

            logger.info("UVDoc distortion correction applied")
            return Image.fromarray(corrected_np)

        except Exception as e:
            logger.warning(f"Distortion correction failed: {e}")
            return image

    # =========================================================================
    # Complete Preprocessing Pipeline
    # =========================================================================

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Complete preprocessing pipeline for a single image.

        Pipeline:
        1. Document detection (YOLO)
        2. Crop to document region
        3. Orientation correction (RapidOCR)
        4. Distortion correction (UVDoc)

        Args:
            image: PIL Image

        Returns:
            Preprocessed PIL Image
        """
        if not self.enable_preprocessing:
            return image

        logger.debug(f"Starting preprocessing for image: {image.size}")
        processed = image

        # Step 1: Document detection and crop
        bbox = self.detect_document(processed)
        if bbox is not None:
            processed = self.crop_document(processed, bbox)
            logger.debug(f"Document detected and cropped: {bbox}")
        else:
            logger.debug("No document detected, using original image")

        # Step 2: Orientation correction
        processed = self.correct_orientation(processed)
        logger.debug("Orientation correction applied")

        # Step 3: Distortion correction
        processed = self.correct_distortion(processed)
        logger.debug("Distortion correction applied")

        logger.debug(f"Preprocessing complete. Final size: {processed.size}")
        return processed

    # =========================================================================
    # Override parent methods to add preprocessing
    # =========================================================================

    def load_pages(
        self, sources: Union[str, bytes, List[Union[str, bytes]]]
    ) -> List[Image.Image]:
        """Load sources with optional preprocessing.

        Args:
            sources: Image sources

        Returns:
            List of PIL Images (preprocessed if enabled)
        """
        pages = super().load_pages(sources)

        if self.enable_preprocessing:
            processed_pages = []
            for page in pages:
                processed_pages.append(self.preprocess_image(page))
            return processed_pages

        return pages

    def load_pages_with_unit_indices(
        self, sources: Union[str, bytes, List[Union[str, bytes]]]
    ) -> Tuple[List[Image.Image], List[int]]:
        """Load sources with unit indices and optional preprocessing.

        Args:
            sources: Image sources

        Returns:
            Tuple of (pages, unit_indices)
        """
        pages, unit_indices = super().load_pages_with_unit_indices(sources)

        if self.enable_preprocessing:
            processed_pages = []
            for page in pages:
                processed_pages.append(self.preprocess_image(page))
            return processed_pages, unit_indices

        return pages, unit_indices

    def iter_pages_with_unit_indices(
        self, sources: Union[str, bytes, List[Union[str, bytes]]]
    ):
        """Iterate pages with unit indices and optional preprocessing.

        Args:
            sources: Image sources

        Yields:
            (page, unit_idx) tuples
        """
        for page, unit_idx in super().iter_pages_with_unit_indices(sources):
            if self.enable_preprocessing:
                page = self.preprocess_image(page)
            yield page, unit_idx
