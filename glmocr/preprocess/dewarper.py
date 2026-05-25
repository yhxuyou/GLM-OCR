"""Dewarping preprocessor.

Corrects curved / warped document surfaces into a flat orthographic view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from glmocr.preprocess.base import BasePreprocessor
from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.config import DewarpConfig

logger = get_logger(__name__)


class Dewarper(BasePreprocessor):
    """Document dewarping (flatten curved surfaces).

    Uses a lightweight generative model (e.g. DocTr, UVDoc, ~30M params,
    ~60 MB VRAM in fp16) to produce a flat, rectified view of the document.

    Thread-safe for concurrent inference when the model is in eval mode.

    Usage::

        dewarper = Dewarper(config)
        dewarper.start()
        flat = dewarper.process(image_bgr)
    """

    def __init__(self, config: "DewarpConfig"):
        super().__init__(config)

        self.model_path: Optional[str] = config.model_path
        self.output_size = config.output_size     # e.g. [1024, 1024]
        self.input_size = config.input_size       # model native input size
        self.enabled: bool = config.enabled
        self.skip_threshold: float = config.skip_threshold

        # Internal state — fill in when you load your model
        self._transform = None   # torchvision transforms

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if not self.enabled:
            logger.info("Dewarper disabled, will pass-through images.")
            return

        if not self.model_path:
            raise ValueError(
                "preprocess.dewarp.model_path is required when "
                "dewarp.enabled is True."
            )

        # --- TODO: replace with your actual model loading ---
        #
        # Example (PyTorch):
        #   import torch
        #
        #   device = self.config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        #   self._model = torch.jit.load(self.model_path).to(device).eval()
        #   self._device = device
        #
        #   self._transform = transforms.Compose([
        #       transforms.ToPILImage(),
        #       transforms.Resize(tuple(self.input_size)),
        #       transforms.ToTensor(),
        #   ])
        # -------------------------------------------------------

        logger.info(
            "Dewarper loaded: model=%s device=%s",
            self.model_path,
            self._device or "CPU",
        )

    def stop(self):
        if self._model is not None:
            del self._model
            self._model = None
        self._transform = None
        self._device = None
        logger.debug("Dewarper stopped.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(
        self, image: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Dewarp the document image.

        Args:
            image: BGR or RGB numpy array (H, W, 3).
            **kwargs: Forwarded for future extensibility.

        Returns:
            Flattened image as a numpy array (H, W, 3).
            If the image is already sufficiently flat
            or ``enabled=False``, returns the original image unchanged.
        """
        if not self.enabled:
            return image

        if self._model is None:
            raise RuntimeError(
                "Dewarper not started. Call start() first."
            )

        # --- TODO: replace with your actual inference pipeline ---
        #
        # Typical steps:
        #   1. Quick warp-detection heuristic (skip if already flat)
        #      if _estimate_warp_score(image) < self.skip_threshold:
        #          return image          # already flat, skip expensive model
        #
        #   2. Preprocess
        #      tensor = self._transform(image).unsqueeze(0).to(self._device)
        #
        #   3. Inference
        #      with torch.no_grad():
        #          flow_or_grid = self._model(tensor)
        #
        #   4. Postprocess: apply the flow field / remap
        #      result = _apply_dewarp_grid(image, flow_or_grid, self.output_size)
        # -------------------------------------------------------

        return image


def _estimate_warp_score(image: np.ndarray) -> float:
    """Heuristic to estimate how warped / curved a document image is.

    Returns a score in [0, 1] where 0 is perfectly flat and 1 is heavily
    warped.  Use this as a fast gate before running the expensive dewarp
    model.

    TODO: implement based on your domain knowledge (line-straightness,
    corner orthogonality, etc.)
    """
    _ = image
    return 0.0