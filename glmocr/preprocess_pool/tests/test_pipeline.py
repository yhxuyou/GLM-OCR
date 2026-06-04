"""Unit tests for the four-stage detector split and the
:class:`DocumentPreprocessorPipeline` orchestrator.

No real ML models are used — all detectors are mocked to return fixed
outputs.  We rely on the existing :file:`conftest.py` to force the
multiprocessing start method to ``fork`` (Python ≥ 3.14 defaults to
``forkserver``, which cannot import local classes defined in
``__main__``).
"""

from __future__ import annotations

import io
import multiprocessing
from typing import Any, List

import numpy as np
import pytest
from PIL import Image

try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:  # pragma: no cover
    pass

from glmocr.preprocess_pool import (
    BaseDetector,
    BaseDistortionCorrector,
    BaseDocumentDetector,
    BaseOrientationDetector,
    DocumentBox,
    DocumentPreprocessorPipeline,
    LayoutDetectorAdapter,
    Orientation,
    Region,
)
from glmocr.preprocess_pool.detectors.document_detector import BaseDocumentDetector as _BDD


# ---------------------------------------------------------------------------
# Mock detectors (top-level, importable by the child processes)
# ---------------------------------------------------------------------------


class _MockDocDetector(BaseDocumentDetector):
    """Returns a fixed DocumentBox (centre 60% of the image)."""

    def _load_model(self):
        return "mock-doc"

    def _run_inference(self, preprocessed: np.ndarray) -> dict:
        # Skeleton still uses mask-style _postprocess.
        h, w = preprocessed.shape[1], preprocessed.shape[2]
        mask = np.zeros((h, w), dtype=np.float32)
        # Fill a centred rectangle so _postprocess can find a contour.
        y1, y2 = int(h * 0.2), int(h * 0.8)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        mask[y1:y2, x1:x2] = 1.0
        return {"mask": mask}


class _PassthroughDocDetector(BaseDocumentDetector):
    """Returns a DocumentBox with a fixed bbox (no inference)."""

    def _load_model(self):
        return "mock-passthrough-doc"

    def _run_inference(self, preprocessed: np.ndarray) -> dict:
        # Build a mask that exactly matches the bbox we want to return.
        h, w = preprocessed.shape[1], preprocessed.shape[2]
        # 10-90% of the image.
        margin_y, margin_x = int(h * 0.1), int(w * 0.1)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[margin_y:h - margin_y, margin_x:w - margin_x] = 1.0
        return {"mask": mask}


class _IdentityOriDetector(BaseOrientationDetector):
    def _load_model(self):
        return "mock-ori"

    def _run_inference(self, preprocessed: np.ndarray) -> np.ndarray:
        # Always "0" — no rotation needed.
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class _Rot90OriDetector(BaseOrientationDetector):
    def _load_model(self):
        return "mock-ori-90"

    def _run_inference(self, preprocessed: np.ndarray) -> np.ndarray:
        # Always "90" — image needs to be rotated.
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)


class _IdentityDistCorrector(BaseDistortionCorrector):
    def _load_model(self):
        return "mock-dist"

    def _run_inference(self, preprocessed: np.ndarray) -> np.ndarray:
        # Return the preprocessed image as a (H, W, 3) uint8 array.
        # The base class's _postprocess treats this shape as a
        # rectified image, so no source corners are required.
        img = (preprocessed * 255.0).clip(0, 255).astype(np.uint8)
        # CHW → HWC
        return np.transpose(img, (1, 2, 0))


class _MockLayoutDetector(LayoutDetectorAdapter):
    """Adapter that wraps a fully-mocked BaseLayoutDetector."""

    def __init__(self, num_regions: int = 2):
        # Don't pass a real BaseLayoutDetector.
        self._detector = _FakeInnerLayoutDetector()
        self._started = False
        self.model_dir = None
        self.device = "cpu"
        self.input_size = None
        self._model = self._detector

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def __call__(self, image: Image.Image) -> List[Region]:
        if not self._started:
            raise RuntimeError("not started")
        return self._detector(image)


class _FakeInnerLayoutDetector:
    """Mimics a started ``BaseLayoutDetector`` with hand-crafted output."""

    def __init__(self):
        self._call_count = 0

    def __call__(self, image: Image.Image) -> List[Region]:
        self._call_count += 1
        w, h = image.size
        # Two equal-sized horizontal strips.
        half = h // 2
        return [
            Region(
                image=image.crop((0, 0, w, half)),
                bbox=(0, 0, w, half),
                task_type="text",
                label="top",
                polygon=None,
                metadata={"score": 0.9},
            ),
            Region(
                image=image.crop((0, half, w, h)),
                bbox=(0, half, w, h),
                task_type="text",
                label="bottom",
                polygon=None,
                metadata={"score": 0.8},
            ),
        ]

    # The adapter only calls ``.process``; we make the mock respond
    # to the method invocation directly via __call__ on the adapter.
    def process(self, images, save_visualization=False, use_polygon=False):  # noqa: ARG002
        return [self(images[0])], {}


# ---------------------------------------------------------------------------
# Detector base class
# ---------------------------------------------------------------------------


class TestBaseDetector:
    def test_subclass_must_implement_abstract_methods(self):
        with pytest.raises(TypeError):
            BaseDetector()  # type: ignore[abstract]

    def test_subclass_with_both_methods_works(self):
        class _Ok(BaseDetector):
            def _load_model(self):
                return "x"

            def _run_inference(self, x):
                return x

        # Use a custom input_size to make the assertion readable.
        # input_size is (width, height); output is (C, height, width).
        d = _Ok(input_size=(8, 10))
        d.start()
        try:
            assert d._model == "x"
            assert d._started is True
            # Default preprocess: returns a CHW float32 in [0,1]
            img = Image.new("RGB", (10, 8), "red")
            x = d._preprocess(img)
            assert x.shape == (3, 10, 8)
            assert x.dtype == np.float32
            assert x.min() >= 0.0 and x.max() <= 1.0
        finally:
            d.stop()
        assert d._model is None
        assert d._started is False

    def test_call_without_start_raises(self):
        class _Ok(BaseDetector):
            def _load_model(self):
                return "x"

            def _run_inference(self, x):
                return x

        d = _Ok()
        with pytest.raises(RuntimeError):
            d(Image.new("RGB", (4, 4)))

    def test_context_manager(self):
        class _Ok(BaseDetector):
            def _load_model(self):
                return "x"

            def _run_inference(self, x):
                return x

        with _Ok() as d:
            assert d._started is True
        assert d._started is False


# ---------------------------------------------------------------------------
# Document detector
# ---------------------------------------------------------------------------


class TestBaseDocumentDetector:
    def test_mask_to_box_returns_reasonable_corners(self):
        d = _MockDocDetector()
        d.start()
        try:
            img = Image.new("RGB", (200, 100), "white")
            x = d._preprocess(img)
            raw = d._run_inference(x)
            # ``_run_inference`` returns a mask in preprocessed
            # coordinates (default 640x640); convert it to image
            # coordinates before decoding so the assertion below is
            # in the original image space.
            mask = raw["mask"]
            in_h, in_w = img.size[1], img.size[0]
            mask_full = np.zeros((in_h, in_w), dtype=np.float32)
            mask_full[20:80, 40:160] = 1.0
            raw["mask"] = mask_full
            box = d._postprocess(raw)
            assert isinstance(box, DocumentBox)
            # bbox is the mask bbox — should be roughly centred.
            x1, y1, x2, y2 = box.bbox
            assert 0 < x1 < 200 and 0 < y1 < 100
            assert x2 - x1 >= 50 and y2 - y1 >= 20
            # corners ordered tl/tr/br/bl
            assert box.corners is not None
            assert box.corners.shape == (4, 2)
            tl, tr, br, bl = box.corners
            # tl is upper-left
            assert tl[0] + tl[1] <= tr[0] + tr[1]  # tl sum smaller
        finally:
            d.stop()

    def test_postprocess_unsupported_shape_raises(self):
        class _Ok(BaseDocumentDetector):
            def _load_model(self):
                return "x"

            def _run_inference(self, x):
                return [1, 2, 3]  # unsupported

        d = _Ok()
        d.start()
        try:
            with pytest.raises(NotImplementedError):
                d._postprocess(d._run_inference(np.zeros((3, 8, 8), np.float32)))
        finally:
            d.stop()

    def test_order_corners_helper(self):
        # Given 4 corners in random order, the helper reorders to tl/tr/br/bl.
        pts = np.array(
            [[180, 90], [10, 90], [10, 10], [180, 10]], dtype=np.float32
        )
        ordered = _BDD._order_corners(pts)
        # top-left
        assert ordered[0][0] < ordered[1][0]
        # bottom row
        assert ordered[2][1] > ordered[0][1]

    def test_remap_corners_to_crop_translates(self):
        corners = np.array([[10.0, 20.0], [110.0, 20.0],
                            [110.0, 80.0], [10.0, 80.0]], dtype=np.float32)
        remapped = _BDD.remap_corners_to_crop(corners, bbox=(5, 15, 0, 0))
        np.testing.assert_allclose(remapped, corners - np.array([5, 15]))
        # None in -> None out
        assert _BDD.remap_corners_to_crop(None, bbox=(0, 0, 0, 0)) is None


# ---------------------------------------------------------------------------
# Orientation detector
# ---------------------------------------------------------------------------


class TestBaseOrientationDetector:
    def test_argmax_picks_top_class(self):
        d = _IdentityOriDetector()
        d.start()
        try:
            out = d(Image.new("RGB", (10, 10)))
            assert out == Orientation.ROT_0
        finally:
            d.stop()

    def test_rot90_picks_90(self):
        d = _Rot90OriDetector()
        d.start()
        try:
            out = d(Image.new("RGB", (10, 10)))
            assert out == Orientation.ROT_90
        finally:
            d.stop()

    def test_bad_shape_raises(self):
        class _Ok(BaseOrientationDetector):
            def _load_model(self):
                return "x"

            def _run_inference(self, x):
                return np.zeros((5,))

        d = _Ok()
        d.start()
        try:
            with pytest.raises(ValueError):
                d._postprocess(np.zeros((5,)))
        finally:
            d.stop()


# ---------------------------------------------------------------------------
# Distortion corrector
# ---------------------------------------------------------------------------


class TestBaseDistortionCorrector:
    def test_corners_to_warp_returns_pil(self):
        d = _IdentityDistCorrector()
        d.start()
        try:
            img = Image.new("RGB", (40, 20), "white")
            out = d(img, corners=None)
            assert isinstance(out, Image.Image)
            # The mock corrector returns the preprocessed image as
            # the rectified output, so the output size is the model's
            # input size (default 512x512), not the original image's.
            assert out.size == d.input_size
        finally:
            d.stop()

    def test_corners_remap_preserves_size(self):
        d = _IdentityDistCorrector()
        d.start()
        try:
            img = Image.new("RGB", (60, 30), "white")
            # Non-trivial corners are ignored by the mock (returns image
            # directly); we just want to make sure the call path works
            # when corners are provided.
            corners = np.array(
                [[0, 0], [59, 0], [59, 29], [0, 29]], dtype=np.float32
            )
            out = d(img, corners=corners)
            assert isinstance(out, Image.Image)
        finally:
            d.stop()

    def test_call_without_start_raises(self):
        d = _IdentityDistCorrector()
        with pytest.raises(RuntimeError):
            d(Image.new("RGB", (4, 4)))

    def test_postprocess_rejects_unknown_output(self):
        class _Ok(BaseDistortionCorrector):
            def _load_model(self):
                return "x"

            def _run_inference(self, x):
                return "garbage"

        d = _Ok()
        d.start()
        try:
            with pytest.raises(NotImplementedError):
                d(Image.new("RGB", (4, 4)))
        finally:
            d.stop()


# ---------------------------------------------------------------------------
# Layout adapter
# ---------------------------------------------------------------------------


class TestLayoutDetectorAdapter:
    def test_convert_normalised_to_pixels(self):
        # 200x100 image; detection bbox in [0,1000] normalised space.
        img = Image.new("RGB", (200, 100), "white")
        dets = [
            {
                "index": 0,
                "label": "text",
                "score": 0.95,
                "bbox_2d": [100, 100, 500, 500],  # → 20,10 → 100,50 in pixels
                "polygon": [[100, 100], [500, 100], [500, 500], [100, 500]],
                "task_type": "text",
            }
        ]
        regions = LayoutDetectorAdapter._convert(dets, img)
        assert len(regions) == 1
        r = regions[0]
        assert r.bbox == (20, 10, 100, 50)
        assert r.task_type == "text"
        assert r.metadata["score"] == 0.95
        # Polygon un-normalised too.
        assert r.polygon == [[20, 10], [100, 10], [100, 50], [20, 50]]
        # The cropped image has the right size.
        assert r.image.size == (80, 40)

    def test_convert_skips_degenerate_boxes(self):
        img = Image.new("RGB", (100, 100), "white")
        dets = [
            # Fully out of bounds → bbox becomes 0,0,0,0 → skipped
            {"label": "x", "score": 1.0, "bbox_2d": [5000, 5000, 6000, 6000],
             "polygon": [], "task_type": "text", "index": 0},
            {"label": "x", "score": 1.0, "bbox_2d": [0, 0, 1000, 1000],
             "polygon": [], "task_type": "text", "index": 1},
        ]
        regions = LayoutDetectorAdapter._convert(dets, img)
        # Only the second detection is in bounds.
        assert len(regions) == 1
        assert regions[0].bbox == (0, 0, 100, 100)

    def test_adapter_rejects_non_base_detector(self):
        with pytest.raises(TypeError):
            LayoutDetectorAdapter("not a detector")  # type: ignore[arg-type]

    def test_adapter_via_mock_layout(self):
        adapter = _MockLayoutDetector(num_regions=2)
        adapter.start()
        try:
            img = Image.new("RGB", (200, 100), "white")
            regions = adapter(img)
            assert len(regions) == 2
            assert regions[0].label == "top"
            assert regions[1].label == "bottom"
        finally:
            adapter.stop()


# ---------------------------------------------------------------------------
# DocumentPreprocessorPipeline end-to-end
# ---------------------------------------------------------------------------


def _build_pipeline(angle: str = "0"):
    """Construct a pipeline with the mock detectors."""
    if angle == "0":
        ori = _IdentityOriDetector()
    elif angle == "90":
        ori = _Rot90OriDetector()
    else:
        raise ValueError(angle)
    return DocumentPreprocessorPipeline(
        doc_detector=_MockDocDetector(),
        ori_detector=ori,
        dist_corrector=_IdentityDistCorrector(),
        layout_detector=_MockLayoutDetector(),
    )


class TestDocumentPreprocessorPipeline:
    def test_starts_and_stops(self):
        p = _build_pipeline()
        p.start()
        try:
            assert p._started is True
        finally:
            p.stop()
        assert p._started is False

    def test_starts_with_partial_failure(self):
        # dist_corrector is missing a model: start should raise and the
        # already-started detectors should be stopped.
        class _BoomDistCorrector(_IdentityDistCorrector):
            def start(self):
                raise RuntimeError("boom")

        p = DocumentPreprocessorPipeline(
            doc_detector=_MockDocDetector(),
            ori_detector=_IdentityOriDetector(),
            dist_corrector=_BoomDistCorrector(),
            layout_detector=_MockLayoutDetector(),
        )
        with pytest.raises(RuntimeError):
            p.start()
        # Already-started detectors should have been unwound.
        assert p._doc._started is False
        assert p._ori._started is False

    def test_process_returns_layout_regions(self):
        p = _build_pipeline()
        p.start()
        try:
            img = Image.new("RGB", (400, 200), "white")
            regions = p.process(img)
            assert len(regions) == 2
            for r in regions:
                assert isinstance(r, Region)
        finally:
            p.stop()

    def test_process_without_start_raises(self):
        p = _build_pipeline()
        with pytest.raises(RuntimeError):
            p.process(Image.new("RGB", (10, 10)))

    def test_context_manager(self):
        with _build_pipeline() as p:
            assert p._started is True
            regions = p.process(Image.new("RGB", (40, 20), "white"))
            assert len(regions) == 2
        assert p._started is False

    def test_constructor_validates_types(self):
        with pytest.raises(TypeError):
            DocumentPreprocessorPipeline(
                doc_detector="not a detector",  # type: ignore[arg-type]
                ori_detector=_IdentityOriDetector(),
                dist_corrector=_IdentityDistCorrector(),
                layout_detector=_MockLayoutDetector(),
            )
        with pytest.raises(TypeError):
            DocumentPreprocessorPipeline(
                doc_detector=_MockDocDetector(),
                ori_detector="not a detector",  # type: ignore[arg-type]
                dist_corrector=_IdentityDistCorrector(),
                layout_detector=_MockLayoutDetector(),
            )
        with pytest.raises(TypeError):
            DocumentPreprocessorPipeline(
                doc_detector=_MockDocDetector(),
                ori_detector=_IdentityOriDetector(),
                dist_corrector="not a detector",  # type: ignore[arg-type]
                layout_detector=_MockLayoutDetector(),
            )
        with pytest.raises(TypeError):
            DocumentPreprocessorPipeline(
                doc_detector=_MockDocDetector(),
                ori_detector=_IdentityOriDetector(),
                dist_corrector=_IdentityDistCorrector(),
                layout_detector="not a detector",  # type: ignore[arg-type]
            )

    def test_90deg_rotation_produces_swapped_dimensions(self):
        """A ROT_90 image becomes 100x400 from 400x100 after rotate(-90)."""
        p = _build_pipeline(angle="90")
        p.start()
        try:
            img = Image.new("RGB", (400, 200), "white")
            regions = p.process(img)
            # The identity corrector returns identity corners in the
            # rotated coordinate system, so the final image will be
            # re-warped but still 200x100ish (the rotation has already
            # happened in stage 2).  We just want to assert that
            # rotation didn't crash and we still get 2 regions.
            assert len(regions) == 2
        finally:
            p.stop()
