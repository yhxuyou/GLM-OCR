"""Preprocessing stage orchestrator.

Coordinates document detection, orientation correction, and dewarping
in sequence.  An optional lightweight lock serialises GPU calls when
multiple pipeline workers share the same preprocessor instances — which
is safe because all models are small and inference is fast.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.config import PipelineConfig
    from glmocr.preprocess.document_detector import DocumentDetector
    from glmocr.preprocess.orientation_corrector import OrientationCorrector
    from glmocr.preprocess.dewarper import Dewarper

logger = get_logger(__name__)


class PreprocessStage:
    """Orchestrates three GPU preprocessors in a fixed pipeline.

    Execution order::

        doc_detector → orientation_corrector → dewarper

    Each step may be disabled in config, in which case the image passes
    through unchanged.  An optional ``use_gpu_lock`` serialises the GPU
    portion when multiple threads compete for the same GPU — this is a
    light-weight lock that only wraps the actual inference calls
    (< 100 ms total), so queue throughput is not meaningfully affected.

    All models are loaded in :meth:`start` and stay resident on GPU
    until :meth:`stop`.  No parameter ping-pong between CPU and GPU.

    Usage::

        stage = PreprocessStage(config)
        stage.start()
        corrected = stage.run(image_bgr)
    """

    def __init__(
        self,
        config: "PipelineConfig",
        use_gpu_lock: bool = False,
    ):
        from glmocr.preprocess.document_detector import DocumentDetector
        from glmocr.preprocess.orientation_corrector import OrientationCorrector
        from glmocr.preprocess.dewarper import Dewarper

        self._config = config

        doc_cfg = getattr(config, "doc_detection", None)
        if doc_cfg is not None:
            self.doc_detector: Optional[DocumentDetector] = DocumentDetector(doc_cfg)
        else:
            self.doc_detector = None

        orient_cfg = getattr(config, "orientation_correction", None)
        if orient_cfg is not None:
            self.orient_corrector: Optional[OrientationCorrector] = (
                OrientationCorrector(orient_cfg)
            )
        else:
            self.orient_corrector = None

        dewarp_cfg = getattr(config, "dewarp", None)
        if dewarp_cfg is not None:
            self.dewarper: Optional[Dewarper] = Dewarper(dewarp_cfg)
        else:
            self.dewarper = None

        self._gpu_lock: Optional[threading.Lock] = (
            threading.Lock() if use_gpu_lock else None
        )
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if self._started:
            return
        logger.info("Starting PreprocessStage...")

        for name, p in self._iter_preprocessors():
            if p is not None:
                p.start()
                logger.debug("  %s: ready (device=%s)", name, p.device)

        self._started = True
        logger.info("PreprocessStage started.")

    def stop(self):
        if not self._started:
            return
        logger.info("Stopping PreprocessStage...")

        for name, p in self._iter_preprocessors():
            if p is not None:
                p.stop()

        self._started = False
        logger.info("PreprocessStage stopped.")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(self, image: np.ndarray) -> np.ndarray:
        """Run the full preprocessing pipeline on a single image.

        Args:
            image: numpy array (H, W, 3), BGR or RGB.

        Returns:
            Corrected image of the same dtype.
        """
        if not self._started:
            raise RuntimeError(
                "PreprocessStage not started. Call start() before run()."
            )

        if self._gpu_lock is not None:
            with self._gpu_lock:
                return self._run_inner(image)
        return self._run_inner(image)

    def _run_inner(self, image: np.ndarray) -> np.ndarray:
        if self.doc_detector is not None:
            image = self.doc_detector.process(image)

        if self.orient_corrector is not None:
            image = self.orient_corrector.process(image)

        if self.dewarper is not None:
            image = self.dewarper.process(image)

        return image

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _iter_preprocessors(self):
        yield "doc_detector", self.doc_detector
        yield "orient_corrector", self.orient_corrector
        yield "dewarper", self.dewarper

    def has_any_enabled(self) -> bool:
        """Return True when at least one preprocessor is enabled."""
        return any(
            p is not None and getattr(p, "enabled", False)
            for _, p in self._iter_preprocessors()
        )