"""Orientation correction preprocessor.

Classifies the image as 0°/90°/180°/270° and rotates it to upright.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from glmocr.preprocess.base import BasePreprocessor
from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.config import OrientationConfig

logger = get_logger(__name__)


class OrientationCorrector(BasePreprocessor):
    """Document orientation detection and rotation correction.

    Uses a lightweight classification model (e.g. ResNet-18, ~11M params,
    ~22 MB VRAM in fp16) to predict the rotation angle and rotates the
    image to upright (0°).

    Thread-safe for concurrent inference when the model is in eval mode.

    Usage::

        corrector = OrientationCorrector(config)
        corrector.start()
        upright = corrector.process(image_bgr)
    """

    # Mapping from class index to rotation degrees (counter-clockwise).
    # Customise if your model uses a different label scheme.
    CLASS_TO_DEGREES = {0: 0, 1: 90, 2: 180, 3: 270}

    def __init__(self, config: "OrientationConfig"):
        super().__init__(config)

        self.model_path: Optional[str] = config.model_path
        self.classes: list = config.classes     # e.g. [0, 90, 180, 270]
        self.min_confidence: float = config.min_confidence
        self.input_size = config.input_size     # e.g. [224, 224]
        self.enabled: bool = config.enabled

        # Internal state — fill in when you load your model
        self._transform = None   # torchvision transforms / preprocessing

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if not self.enabled:
            logger.info("OrientationCorrector disabled, will pass-through images.")
            return

        if not self.model_path:
            raise ValueError(
                "preprocess.orientation_correction.model_path is required "
                "when orientation_correction.enabled is True."
            )

        # --- TODO: replace with your actual model loading ---
        #
        # Example (PyTorch):
        #   import torch
        #   from torchvision import transforms
        #
        #   device = self.config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        #   self._model = torch.jit.load(self.model_path).to(device).eval()
        #   self._device = device
        #
        #   self._transform = transforms.Compose([
        #       transforms.ToPILImage(),
        #       transforms.Resize(tuple(self.input_size)),
        #       transforms.ToTensor(),
        #       transforms.Normalize(
        #           mean=[0.485, 0.456, 0.406],
        #           std=[0.229, 0.224, 0.225],
        #       ),
        #   ])
        # -------------------------------------------------------

        logger.info(
            "OrientationCorrector loaded: model=%s device=%s",
            self.model_path,
            self._device or "CPU",
        )

    def stop(self):
        if self._model is not None:
            del self._model
            self._model = None
        self._transform = None
        self._device = None
        logger.debug("OrientationCorrector stopped.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(
        self, image: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Detect orientation and rotate to upright.

        Args:
            image: BGR or RGB numpy array (H, W, 3).
            **kwargs: Forwarded for future extensibility.

        Returns:
            Upright image as a numpy array of the same dtype.
            If orientation is already 0° with high confidence or
            ``enabled=False``, returns the original image unchanged.
        """
        if not self.enabled:
            return image

        if self._model is None:
            raise RuntimeError(
                "OrientationCorrector not started. Call start() first."
            )

        # --- TODO: replace with your actual inference pipeline ---
        #
        # Typical steps:
        #   1. Preprocess
        #      tensor = self._transform(image).unsqueeze(0).to(self._device)
        #
        #   2. Inference
        #      with torch.no_grad():
        #          logits = self._model(tensor)
        #      probs = torch.softmax(logits, dim=-1)
        #      class_idx = probs.argmax(dim=-1).item()
        #      confidence = probs.max().item()
        #
        #   3. Rotate if needed
        #      degrees = self.CLASS_TO_DEGREES.get(class_idx, 0)
        #      if degrees == 0 or confidence < self.min_confidence:
        #          return image
        #      return _rotate_image(image, degrees)
        # -------------------------------------------------------

        return image


def _rotate_image(image: np.ndarray, degrees_ccw: int) -> np.ndarray:
    """Rotate an image by multiples of 90° (counter-clockwise).

    Args:
        image: numpy array (H, W, C).
        degrees_ccw: 90, 180, or 270.

    Returns:
        Rotated array.
    """
    import cv2

    if degrees_ccw == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif degrees_ccw == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif degrees_ccw == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image