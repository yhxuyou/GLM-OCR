"""Document detection preprocessor.

Detects the document region within an image and crops to it, removing
background / deskewing artefacts that would confuse the layout detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from glmocr.preprocess.base import BasePreprocessor
from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.config import DocDetectionConfig

logger = get_logger(__name__)


class DocumentDetector(BasePreprocessor):
    """Document boundary detection and cropping.

    Uses a lightweight object-detection model to locate the document
    quadrilateral within the input image.  Output is a cropped and
    perspective-corrected image containing only the document.

    Typical models: YOLO-v8n, Rt-DETR-R18, or any ONNX detection model
    with ~3M parameters (~6 MB VRAM in fp16).

    Thread-safe for concurrent inference when the model is in eval mode.

    Usage::

        detector = DocumentDetector(config)
        detector.start()
        cropped = detector.process(image_bgr)
    """

    def __init__(self, config: "DocDetectionConfig"):
        super().__init__(config)

        self.model_path: Optional[str] = config.model_path
        self.confidence_threshold: float = config.confidence_threshold
        self.input_size = config.input_size          # e.g. [640, 640]
        self.padding: float = config.padding          # expand crop margin
        self.enabled: bool = config.enabled

        # Internal state — fill in when you load your model
        self._preprocessor = None   # image preprocessor (resize/normalise)
        self._postprocessor = None  # NMS / bbox decoder

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if not self.enabled:
            logger.info("DocumentDetector disabled, will pass-through images.")
            return

        if not self.model_path:
            raise ValueError(
                "preprocess.doc_detection.model_path is required when "
                "doc_detection.enabled is True."
            )

        # --- TODO: replace with your actual model loading ---
        #
        # Example (ONNX Runtime):
        #   import onnxruntime as ort
        #   self._model = ort.InferenceSession(
        #       self.model_path,
        #       providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        #   )
        #
        # Example (PyTorch):
        #   device = self.config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        #   self._model = torch.jit.load(self.model_path).to(device).eval()
        #   self._device = device
        #
        # Example (OpenCV DNN):
        #   self._model = cv2.dnn.readNetFromONNX(self.model_path)
        #   self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #   self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # -------------------------------------------------------

        logger.info(
            "DocumentDetector loaded: model=%s device=%s",
            self.model_path,
            self._device or "CPU",
        )

    def stop(self):
        if self._model is not None:
            del self._model
            self._model = None
        self._preprocessor = None
        self._postprocessor = None
        self._device = None
        logger.debug("DocumentDetector stopped.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(
        self, image: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Detect document region and crop.

        Args:
            image: BGR numpy array (H, W, 3).
            **kwargs: Forwarded for future extensibility.

        Returns:
            Cropped and perspective-corrected BGR numpy array.
            If no document is found or ``enabled=False``, returns the
            original image unchanged.
        """
        if not self.enabled:
            return image

        if self._model is None:
            raise RuntimeError(
                "DocumentDetector not started. Call start() first."
            )

        # --- TODO: replace with your actual inference pipeline ---
        #
        # Typical steps:
        #   1. Preprocess: resize to self.input_size, normalise
        #      blob = self._preprocessor(image)
        #
        #   2. Inference
        #      outputs = self._model(blob)
        #
        #   3. Postprocess: decode boxes, apply NMS, filter by threshold
        #      boxes, scores = self._postprocessor(outputs)
        #
        #   4. Find best document candidate (largest area, highest score)
        #      doc_corners = _find_best_document(boxes, scores,
        #                                        self.confidence_threshold)
        #
        #   5. Perspective-correct
        #      if doc_corners is not None:
        #          image = _four_point_transform(image, doc_corners)
        # -------------------------------------------------------

        return image


def _find_best_document(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Optional[np.ndarray]:
    """Select the most likely document quadrilateral from detections.

    Args:
        boxes: (N, 4, 2) array of corner points.
        scores: (N,) confidence scores.
        threshold: Minimum score to consider.

    Returns:
        (4, 2) array of corners, or ``None`` if nothing found.
    """
    mask = scores >= threshold
    if not mask.any():
        return None
    idx = scores[mask].argmax()
    return boxes[mask][idx]