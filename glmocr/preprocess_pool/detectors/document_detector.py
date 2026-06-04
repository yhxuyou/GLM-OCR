"""Stage 1 detector: find the document region in a photo.

Subclass :class:`BaseDocumentDetector` and implement ``_load_model`` and
``_run_inference``.  The default ``_postprocess`` expects a
``{"mask": (H, W) float32 array}`` dict from your model — if your model
returns a different shape (e.g. keypoints + boxes), override
``_postprocess`` instead.

The classical-CV ``_mask_to_document_box`` helper does the standard
contour extraction / 4-corner approximation that is shared by almost
all segmentation-based document detectors.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from .base import BaseDetector, DocumentBox


class BaseDocumentDetector(BaseDetector):
    """Abstract base for "find the document in the image" detectors.

    Example subclass that returns a soft mask from ONNX Runtime::

        import onnxruntime as ort
        from PIL import Image

        class MyDocDetector(BaseDocumentDetector):
            def _load_model(self):
                return ort.InferenceSession(
                    f"{self.model_dir}/doc_seg.onnx",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )

            def _run_inference(self, x):
                # x: (C, H, W) float32 in [0, 1] from default _preprocess
                x = x[None]                              # add batch
                mask = self._model.run(None, {"input": x})[0][0, 0]
                return {"mask": mask}
    """

    def __init__(
        self,
        mask_threshold: float = 0.5,
        contour_epsilon_ratio: float = 0.02,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not 0.0 < mask_threshold < 1.0:
            raise ValueError("mask_threshold must be in (0, 1)")
        self.mask_threshold = float(mask_threshold)
        # Fraction of contour perimeter used by approxPolyDP.  Smaller
        # values follow the contour more closely; larger values
        # approximate to fewer vertices.
        self.contour_epsilon_ratio = float(contour_epsilon_ratio)

    # ---- Default postprocess for mask-style models ----

    def _postprocess(self, raw: Any) -> DocumentBox:
        """Decode a model output dict into a :class:`DocumentBox`.

        Default behaviour: expects ``raw`` to be a dict with a
        ``"mask"`` key containing a ``(H, W)`` float32 segmentation
        probability map.  Override for other output shapes.
        """
        if isinstance(raw, dict) and "mask" in raw:
            mask = np.asarray(raw["mask"], dtype=np.float32)
            return self._mask_to_document_box(mask)
        raise NotImplementedError(
            f"{type(self).__name__}._postprocess expects dict with 'mask'; "
            "override _postprocess to support other model output shapes "
            "(boxes, keypoints, polygons, ...)."
        )

    # ---- Classical CV helper: mask → 4 corners + bbox ----

    def _mask_to_document_box(self, mask: np.ndarray) -> DocumentBox:
        """Threshold the mask, find the largest contour, approximate
        it to 4 corners, and return a :class:`DocumentBox`."""
        if mask.ndim != 2:
            raise ValueError(
                f"_mask_to_document_box expects a 2D mask, got shape {mask.shape}"
            )
        h, w = mask.shape
        bin_mask = (mask > self.mask_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return DocumentBox(bbox=(0, 0, w, h), corners=None, confidence=0.0)
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 4:
            return DocumentBox(bbox=(0, 0, w, h), corners=None, confidence=0.0)

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, self.contour_epsilon_ratio * peri, closed=True
        )

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            # Fall back to the minimum-area rotated rectangle when
            # the contour doesn't simplify to exactly 4 vertices
            # (e.g. documents with torn or curved edges).
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect).astype(np.float32)

        corners = self._order_corners(corners)
        x1, y1 = np.floor(corners.min(axis=0)).astype(int)
        x2, y2 = np.ceil(corners.max(axis=0)).astype(int)
        # Clip to image bounds.
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return DocumentBox(bbox=(0, 0, w, h), corners=None, confidence=0.0)

        # Estimate confidence as contour area / bbox area (IoU-ish).
        bbox_area = float((x2 - x1) * (y2 - y1))
        conf = float(cv2.contourArea(contour)) / max(bbox_area, 1.0)
        conf = float(min(1.0, max(0.0, conf)))

        return DocumentBox(
            bbox=(x1, y1, x2, y2),
            corners=corners,
            confidence=conf,
        )

    # ---- Helpers ----

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """Order four points as ``[top-left, top-right, bottom-right, bottom-left]``."""
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape != (4, 2):
            raise ValueError(f"Expected (4, 2) array, got {pts.shape}")
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

    @staticmethod
    def remap_corners_to_crop(
        corners: Optional[np.ndarray], bbox: tuple
    ) -> Optional[np.ndarray]:
        """Translate corners from the original image to the cropped
        coordinate system defined by ``bbox``.

        Returns ``None`` if ``corners`` is ``None`` (no corners to
        translate).
        """
        if corners is None:
            return None
        x1, y1, _, _ = bbox
        return corners - np.array([x1, y1], dtype=np.float32)


__all__ = ["BaseDocumentDetector"]
