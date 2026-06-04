"""Stage 3 detector: perspective / distortion correction.

Unlike the other detectors, this one takes **two** inputs — an image
*and* the four corners of the document region (from stage 1).  Its
``__call__`` signature is therefore ``(image, corners=None)``.

Subclasses implement the two abstract methods as usual.  The default
``__call__`` is intentionally **not** a thin wrapper around
``BaseDetector.__call__`` because of the extra ``corners`` argument.

The default postprocess handles two common model output shapes:

* ``(H, W, 3)`` ``uint8`` array — already a rectified image, returned
  as-is.
* ``(4, 2)`` float array — destination corners in the same coordinate
  system as the input image.  We compute a homography from
  ``corners → new_corners`` and ``cv2.warpPerspective`` the image.

If your model returns something else (e.g. a flow field), override
``_postprocess``.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image

from .base import BaseDetector


class BaseDistortionCorrector(BaseDetector):
    """Abstract base for "straighten a distorted document" detectors.

    Example model-based subclass::

        class MyDistCorrector(BaseDistortionCorrector):
            def _load_model(self):
                import onnxruntime as ort
                return ort.InferenceSession(
                    f"{self.model_dir}/unwarp.onnx",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )

            def _run_inference(self, x):
                # x: (C, H, W) float32 in [0, 1]
                return self._model.run(None, {"input": x[None]})[0][0]
                # Could be (H, W, 3) rectified image, or (4, 2) dst corners
    """

    # Use a moderate input size by default; can be overridden.
    DEFAULT_INPUT_SIZE = (512, 512)

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("input_size", self.DEFAULT_INPUT_SIZE)
        super().__init__(**kwargs)

    # ---- Overridden __call__ to accept the extra ``corners`` arg ----

    def __call__(
        self,
        image: Image.Image,
        corners: Optional[np.ndarray] = None,
    ) -> Image.Image:
        """Run the full preprocessing → inference → warping pipeline.

        Args:
            image: Cropped, orientation-corrected PIL image.
            corners: ``(4, 2)`` array of the four corners of the
                document within ``image``, in the order produced by
                :class:`BaseDocumentDetector`.  ``None`` means the
                corrector has to find them itself (e.g. by contour
                detection on a preprocessed mask).

        Returns:
            The straightened, rectified PIL image.
        """
        if not self._started:
            raise RuntimeError(
                f"{type(self).__name__} is not started; call .start() first."
            )
        x = self._preprocess(image)
        raw = self._run_inference(x)
        return self._postprocess(raw, image, corners)

    # ---- Default postprocess: accept either image or new corners ----

    def _postprocess(
        self,
        raw: Any,
        image: Image.Image,
        corners: Optional[np.ndarray],
    ) -> Image.Image:
        """Decode model output into a rectified PIL image.

        Default behaviour:

        * If ``raw`` is a ``(H, W, 3)`` ``uint8`` array, treat it as
          the rectified image and return it.
        * If ``raw`` is a ``(4, 2)`` array and ``corners`` is not
          ``None``, compute a homography from ``corners`` to ``raw``
          and warp ``image``.

        Subclasses with different output shapes (e.g. flow fields) must
        override this method.
        """
        arr = np.asarray(raw)
        # Case 1: rectified image (H, W, 3)
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        # Case 2: 4 destination corners
        if arr.ndim == 2 and arr.shape == (4, 2):
            if corners is None:
                raise ValueError(
                    "Corrector returned new corners but no source corners "
                    "were provided to __call__().  Pass corners=... or "
                    "override _postprocess."
                )
            return self._warp(image, corners, arr.astype(np.float32))
        raise NotImplementedError(
            f"{type(self).__name__}._postprocess could not interpret model "
            f"output with shape {arr.shape} and dtype {arr.dtype}.  Override "
            "_postprocess to support this output."
        )

    # ---- Classical helper: perspective warp ----

    @staticmethod
    def _warp(
        image: Image.Image,
        src_corners: np.ndarray,
        dst_corners: np.ndarray,
        output_size: Optional[tuple] = None,
    ) -> Image.Image:
        """Compute homography ``src → dst`` and warp ``image``.

        ``output_size`` defaults to ``image.size``.
        """
        if src_corners.shape != (4, 2) or dst_corners.shape != (4, 2):
            raise ValueError(
                "_warp expects (4, 2) source and destination corner arrays"
            )
        M, _ = cv2.findHomography(
            src_corners.astype(np.float32),
            dst_corners.astype(np.float32),
            method=cv2.RANSAC,
        )
        if M is None:
            # Fall back to a non-robust method if RANSAC fails.
            M = cv2.getPerspectiveTransform(
                src_corners.astype(np.float32),
                dst_corners.astype(np.float32),
            )
        w, h = output_size if output_size is not None else image.size
        warped = cv2.warpPerspective(np.asarray(image), M, (w, h))
        return Image.fromarray(warped)


__all__ = ["BaseDistortionCorrector"]
