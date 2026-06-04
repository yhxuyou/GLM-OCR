"""Stage 4 adapter: wrap any :class:`BaseLayoutDetector` (including the
SDK's :class:`PPDocLayoutDetector`) so it conforms to the
:class:`BaseDetector` interface and produces :class:`Region` objects.

Unlike the other three detectors, this one is **not abstract** — it is
a fully working adapter.  Users just pass in a configured
``BaseLayoutDetector`` and the adapter takes care of:

* The lifecycle (``start`` / ``stop`` delegate to the inner detector).
* Coordinate conversion (the SDK returns ``[0, 1000]``-normalised
  coordinates; the adapter turns them back into pixels and crops the
  image for each region).
* Polygon unnormalisation.
"""

from __future__ import annotations

from typing import Any, List

from PIL import Image

from glmocr.layout.base import BaseLayoutDetector
from glmocr.preprocess_pool.protocols import Region

from .base import BaseDetector


class LayoutDetectorAdapter(BaseDetector):
    """Adapter that lets any :class:`BaseLayoutDetector` plug into
    :class:`DocumentPreprocessorPipeline`.

    Args:
        base_detector: An **un-started** instance of
            :class:`BaseLayoutDetector` (e.g. ``PPDocLayoutDetector(cfg.pipeline.layout)``).

    Example::

        from glmocr.layout import PPDocLayoutDetector
        from glmocr.config import load_config
        from glmocr.preprocess_pool.detectors import LayoutDetectorAdapter

        cfg = load_config()
        adapter = LayoutDetectorAdapter(PPDocLayoutDetector(cfg.pipeline.layout))
        # ... pass adapter to DocumentPreprocessorPipeline
    """

    # The adapter doesn't need its own model_dir / device / input_size;
    # those are owned by the wrapped BaseLayoutDetector.  We override
    # __init__ to accept *only* the inner detector.
    def __init__(self, base_detector: BaseLayoutDetector, **kwargs: Any):
        # Skip BaseDetector.__init__ deliberately — we don't store
        # model_dir / device / input_size / _model.
        if not isinstance(base_detector, BaseLayoutDetector):
            raise TypeError(
                f"base_detector must be a BaseLayoutDetector, got "
                f"{type(base_detector).__name__}"
            )
        self._detector = base_detector
        self._started = False
        # Mimic BaseDetector's external state for isinstance checks.
        self.model_dir = getattr(base_detector, "model_dir", None)
        self.device = getattr(base_detector, "_device", None)
        self.input_size = None
        self._model = base_detector  # for protocol compatibility

    # ---- Lifecycle: delegate to the inner detector ----

    def start(self) -> None:
        if self._started:
            return
        self._detector.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self._detector.stop()
        finally:
            self._started = False

    # ---- BaseDetector's abstract stubs ----
    # (We override __call__ below so we never actually need these.)

    def _load_model(self) -> Any:  # pragma: no cover - never called
        return self._detector

    def _run_inference(self, preprocessed: Any) -> Any:  # pragma: no cover
        raise NotImplementedError(
            "LayoutDetectorAdapter uses the wrapped BaseLayoutDetector's "
            "process() directly; do not call _run_inference() on it."
        )

    def _preprocess(self, image: Image.Image) -> Any:  # pragma: no cover
        return image

    def _postprocess(self, raw: Any) -> Any:  # pragma: no cover
        return raw

    # ---- Real entry point ----

    def __call__(self, image: Image.Image) -> List[Region]:
        """Run the wrapped layout detector and return ``Region`` objects.

        Coordinates are converted from the SDK's ``[0, 1000]`` normalised
        space back to pixels, the image is cropped for each region, and
        polygons are also un-normalised.
        """
        if not self._started:
            raise RuntimeError(
                "LayoutDetectorAdapter is not started; call .start() first."
            )
        all_results, _ = self._detector.process(
            images=[image], save_visualization=False, use_polygon=False
        )
        return self._convert(all_results[0], image)

    # ---- Conversion helper ----

    @staticmethod
    def _convert(detections: list, image: Image.Image) -> List[Region]:
        """Convert SDK detection dicts to :class:`Region` objects.

        The SDK format (per detection) is::

            {
                "index": int,
                "label": str,
                "score": float,
                "bbox_2d": [x1, y1, x2, y2],  # 0-1000 normalised
                "polygon": [[x, y], ...],     # 0-1000 normalised
                "task_type": str,             # "text" / "table" / "formula"
            }
        """
        w, h = image.size
        out: List[Region] = []
        for d in detections:
            x1n, y1n, x2n, y2n = d["bbox_2d"]
            x1 = int(round(x1n * w / 1000.0))
            y1 = int(round(y1n * h / 1000.0))
            x2 = int(round(x2n * w / 1000.0))
            y2 = int(round(y2n * h / 1000.0))
            # Clip to image bounds.
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                # Degenerate box: skip this detection.
                continue
            crop = image.crop((x1, y1, x2, y2))
            polygon = [
                [int(round(px * w / 1000.0)), int(round(py * h / 1000.0))]
                for px, py in d.get("polygon", [])
            ]
            out.append(
                Region(
                    image=crop,
                    bbox=(x1, y1, x2, y2),
                    task_type=d.get("task_type", "text"),
                    label=d.get("label", ""),
                    polygon=polygon if polygon else None,
                    metadata={
                        "score": float(d.get("score", 0.0)),
                        "index": int(d.get("index", 0)),
                    },
                )
            )
        return out


__all__ = ["LayoutDetectorAdapter"]
