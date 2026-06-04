"""Detector base classes and shared data types.

All four preprocessing stages (document detection, orientation detection,
distortion correction, layout detection) share the same lifecycle:

* ``__init__`` — store configuration
* ``start()``  — load model and warm up
* ``__call__(image)`` — preprocess → infer → postprocess
* ``stop()``   — release resources

Subclasses fill in two abstract methods::

    def _load_model(self) -> Any:           # load weights from self.model_dir
    def _run_inference(self, x) -> Any:    # model forward

Default ``_preprocess`` / ``_postprocess`` use common classical-CV
recipes (resize + CHW normalize, argmax, contour extraction, etc.).
Override them when your model needs something specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------


class Orientation(IntEnum):
    """Document rotation in degrees, clockwise.

    Semantics: the value is "the rotation that the document was captured
    in relative to the upright reading orientation".  When ``ROT_90`` is
    returned, the document is currently rotated 90° clockwise from
    upright, and the pipeline will apply ``image.rotate(-90, expand=True)``
    to bring it back.
    """

    ROT_0 = 0
    ROT_90 = 90
    ROT_180 = 180
    ROT_270 = 270


@dataclass
class DocumentBox:
    """Output of the document-detection stage.

    Attributes:
        bbox: ``(x1, y1, x2, y2)`` in the **original** image pixel space.
        corners: ``(4, 2)`` float32 array of the four corners in the
            original image coordinate system.  Order is
            ``[top-left, top-right, bottom-right, bottom-left]``.
        confidence: Detection confidence in ``[0, 1]``.
        metadata: Free-form user metadata.
    """

    bbox: Tuple[int, int, int, int]
    corners: Optional[np.ndarray] = None
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Common base class
# ---------------------------------------------------------------------------


class BaseDetector(ABC):
    """Lifecycle and skeleton shared by the four detector classes.

    The four stages differ only in:
      * what their ``__call__`` returns (an image, an int, a list of
        regions, etc.),
      * the contract of ``_run_inference``'s raw output,
      * the decoding logic in ``_postprocess``.

    Everything else (model loading, start/stop, common preprocessing) is
    inherited.

    Subclasses **must** implement the two abstract methods below; the
    other hooks have sensible defaults.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cpu",
        input_size: Tuple[int, int] = (640, 640),
        **kwargs: Any,
    ):
        """Store configuration; the model is loaded lazily by ``start()``.

        Args:
            model_dir: Local directory or HF id pointing at the weights.
            device: Target device string (``"cpu"``, ``"cuda"``,
                ``"cuda:0"``).  How it is consumed is up to the
                subclass's ``_load_model``.
            input_size: ``(width, height)`` of the model input.  Used by
                the default ``_preprocess``.
            **kwargs: Forwarded to the subclass.
        """
        self.model_dir = model_dir
        self.device = device
        self.input_size = input_size
        self._model: Any = None
        self._started = False

    # ---- Abstract extension points (subclasses MUST override) ----

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the model from ``self.model_dir`` and return it.

        The returned object is stored in ``self._model`` and is what
        ``_run_inference`` will receive as ``self._model``.
        """

    @abstractmethod
    def _run_inference(self, preprocessed: np.ndarray) -> Any:
        """Run the model forward pass and return the raw output.

        Args:
            preprocessed: The output of ``_preprocess`` (default is
                ``(C, H, W)`` float32 in ``[0, 1]``).
        """

    # ---- Default preprocessing (override only when your model needs
    #      something different) ----

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize → float32 in [0, 1] → CHW.

        This is the most common preprocessing recipe for image
        classifiers / detection heads.  Subclasses working with BGR,
        ``[-1, 1]`` normalisation, or different layouts should override.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        img = image.resize(self.input_size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # HWC → CHW
        return np.transpose(arr, (2, 0, 1))

    def _postprocess(self, raw: Any) -> Any:
        """Default no-op postprocessing.

        Subclasses that decode the model output (NMS, argmax, etc.) must
        override this.
        """
        return raw

    # ---- Lifecycle hooks (override only when truly necessary) ----

    def _on_start(self) -> None:
        """Subclass hook called after the model is loaded.

        Use this for warmup, binding the model to a specific device, or
        building a JIT session.  Default: do nothing.
        """

    def _on_stop(self) -> None:
        """Subclass hook called before ``self._model`` is cleared.

        Use this for releasing CUDA memory, closing sessions, etc.
        Default: do nothing.
        """

    def start(self) -> None:
        """Load the model once.  Idempotent."""
        if self._started:
            return
        self._model = self._load_model()
        try:
            self._on_start()
        except Exception:  # pragma: no cover - defensive
            self._model = None
            raise
        self._started = True

    def stop(self) -> None:
        """Release model resources.  Idempotent."""
        if not self._started:
            return
        try:
            self._on_stop()
        finally:
            self._model = None
            self._started = False

    # ---- Default entry point ----

    def __call__(self, image: Image.Image) -> Any:
        """Run the full preprocess → infer → postprocess cycle.

        Subclasses that need to pass extra arguments to inference
        (e.g. document corners for distortion correction) override this
        method.
        """
        if not self._started:
            raise RuntimeError(
                f"{type(self).__name__} is not started; call .start() first."
            )
        x = self._preprocess(image)
        raw = self._run_inference(x)
        return self._postprocess(raw)

    # ---- Convenience ----

    def __enter__(self) -> "BaseDetector":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


__all__ = ["BaseDetector", "DocumentBox", "Orientation"]
