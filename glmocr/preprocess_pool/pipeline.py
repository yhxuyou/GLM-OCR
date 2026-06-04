"""Composite document preprocessor that orchestrates the four stages.

:mod:`glmocr.preprocess_pool.detectors` provides the four independent
detectors.  This module glues them together in the standard order
document → orientation → distortion → layout.

The pipeline **is itself a** :class:`DocumentPreprocessor`, so it can
be passed to :class:`PreprocessPool` directly.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from PIL import Image

from glmocr.preprocess_pool.detectors.base import Orientation
from glmocr.preprocess_pool.detectors.distortion_corrector import (
    BaseDistortionCorrector,
)
from glmocr.preprocess_pool.detectors.document_detector import BaseDocumentDetector
from glmocr.preprocess_pool.detectors.layout_adapter import LayoutDetectorAdapter
from glmocr.preprocess_pool.detectors.orientation_detector import (
    BaseOrientationDetector,
)
from glmocr.preprocess_pool.protocols import DocumentPreprocessor, Region
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentPreprocessorPipeline(DocumentPreprocessor):
    """Wire the four detectors into a standard preprocessing pipeline.

    The flow is::

        ┌─────────────────┐
        │ DocumentDetector│  find the document, get bbox + 4 corners
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │OrientationDet.  │  classify 0/90/180/270
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │DistortionCorr.  │  perspective-warp the document flat
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ LayoutDetector  │  run PP-DocLayoutV3 (or custom) → Regions
        └─────────────────┘

    Args:
        doc_detector:   Stage 1.
        ori_detector:   Stage 2.
        dist_corrector: Stage 3.
        layout_detector: Stage 4 (typically a :class:`LayoutDetectorAdapter`).

    Example::

        cfg = GlmOcrConfig.from_env()
        pipeline = DocumentPreprocessorPipeline(
            doc_detector=MyDocDetector(model_dir="/m/doc"),
            ori_detector=MyOriDetector(model_dir="/m/ori"),
            dist_corrector=MyDistCorrector(model_dir="/m/dist"),
            layout_detector=LayoutDetectorAdapter(
                PPDocLayoutDetector(cfg.pipeline.layout)
            ),
        )
        pool = PreprocessPool(
            preprocessor=lambda: _build_pipeline(cfg),  # 4 workers, each loads its own
            config=cfg,
        )
    """

    def __init__(
        self,
        doc_detector: BaseDocumentDetector,
        ori_detector: BaseOrientationDetector,
        dist_corrector: BaseDistortionCorrector,
        layout_detector: LayoutDetectorAdapter,
    ):
        if not isinstance(doc_detector, BaseDocumentDetector):
            raise TypeError(
                f"doc_detector must be a BaseDocumentDetector, got "
                f"{type(doc_detector).__name__}"
            )
        if not isinstance(ori_detector, BaseOrientationDetector):
            raise TypeError(
                f"ori_detector must be a BaseOrientationDetector, got "
                f"{type(ori_detector).__name__}"
            )
        if not isinstance(dist_corrector, BaseDistortionCorrector):
            raise TypeError(
                f"dist_corrector must be a BaseDistortionCorrector, got "
                f"{type(dist_corrector).__name__}"
            )
        if not isinstance(layout_detector, LayoutDetectorAdapter):
            raise TypeError(
                f"layout_detector must be a LayoutDetectorAdapter, got "
                f"{type(layout_detector).__name__}"
            )
        self._doc = doc_detector
        self._ori = ori_detector
        self._dist = dist_corrector
        self._lay = layout_detector
        self._started = False

    # ---- Lifecycle ----

    def start(self) -> None:
        if self._started:
            return
        self._doc.start()
        try:
            self._ori.start()
        except Exception:
            self._safe_stop(self._doc)
            raise
        try:
            self._dist.start()
        except Exception:
            self._safe_stop(self._ori)
            self._safe_stop(self._doc)
            raise
        try:
            self._lay.start()
        except Exception:
            self._safe_stop(self._dist)
            self._safe_stop(self._ori)
            self._safe_stop(self._doc)
            raise
        self._started = True
        logger.info("DocumentPreprocessorPipeline started")

    def stop(self) -> None:
        if not self._started:
            return
        # Best-effort shutdown: stop in reverse order.
        for det in (self._lay, self._dist, self._ori, self._doc):
            self._safe_stop(det)
        self._started = False
        logger.info("DocumentPreprocessorPipeline stopped")

    @staticmethod
    def _safe_stop(det: Any) -> None:
        try:
            det.stop()
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Error stopping %s: %s", type(det).__name__, e)

    # ---- The actual preprocessing pipeline ----

    def process(
        self, image: Image.Image, **kwargs: Any
    ) -> List[Region]:
        """Run document detection → orientation → distortion → layout.

        Args:
            image: Input PIL image (full page, possibly a photo).
            **kwargs: Ignored.  Reserved for future hooks (e.g. page
                number, source document id).

        Returns:
            A list of :class:`Region` objects ready for the OCR stage.
        """
        if not self._started:
            raise RuntimeError(
                "DocumentPreprocessorPipeline is not started; call .start() first."
            )

        # ---- Stage 1: document detection ----
        box = self._doc(image)
        x1, y1, x2, y2 = box.bbox
        cropped = image.crop((x1, y1, x2, y2))

        # ---- Stage 2: orientation ----
        angle = self._ori(cropped)
        if int(angle) != 0:
            rotated = cropped.rotate(-int(angle), expand=True)
        else:
            rotated = cropped

        # ---- Stage 3: distortion correction ----
        # The 4 corners returned by stage 1 are in the original image's
        # coordinate system.  After stage 1's crop, the corners have to
        # be re-expressed in the cropped image's coordinate system.
        # After stage 2's rotation they are no longer valid (the crop
        # has different dimensions), so we pass ``None`` to the
        # corrector and let it find new corners on its own.
        if int(angle) == 0 and box.corners is not None:
            corners = box.corners - np.array(
                [box.bbox[0], box.bbox[1]], dtype=np.float32
            )
        else:
            corners = None
        corrected = self._dist(rotated, corners=corners)

        # ---- Stage 4: layout detection ----
        regions = self._lay(corrected)
        logger.debug(
            "pipeline produced %d regions (doc_box=%s, angle=%s)",
            len(regions),
            box.bbox,
            angle,
        )
        return regions

    # ---- Convenience ----

    def __enter__(self) -> "DocumentPreprocessorPipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


__all__ = ["DocumentPreprocessorPipeline"]
