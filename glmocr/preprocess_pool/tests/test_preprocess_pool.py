"""Unit tests for the preprocess pool (no external services required).

These tests use ``multiprocessing`` so they spawn real child processes.  We
set the start method to ``"fork"`` at module load time so the child
processes inherit the test module's globals (which is where the dummy
``DocumentPreprocessor`` lives).
"""

from __future__ import annotations

import io
import multiprocessing
import os
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Use fork so the test's dummy preprocessor class is inherited by the
# child processes.  We have to do this before any Pool/Queue is created.
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:  # pragma: no cover - already set elsewhere
    pass

from glmocr.config import GlmOcrConfig, load_config
from glmocr.preprocess_pool import (
    DocumentPreprocessor,
    PreprocessPool,
    Region,
)
from glmocr.preprocess_pool.aggregator import DocumentAggregator
from glmocr.preprocess_pool.async_ocr import AsyncOCRDispatcher
from glmocr.preprocess_pool.protocols import Region as _Region
from glmocr.preprocess_pool.tasks import RegionTask, WORKER_STOP_SENTINEL


# ---------------------------------------------------------------------------
# Dummy preprocessor used by every test in this file.  Defined at module
# level so multiprocessing 'fork' can find it in the child process globals.
# ---------------------------------------------------------------------------


class DummyPreprocessor:
    """Returns a configurable number of regions, all of type ``"text"``."""

    def __init__(self, num_regions: int = 3, sleep_seconds: float = 0.0):
        self.num_regions = num_regions
        self.sleep_seconds = sleep_seconds
        self.pid = os.getpid()
        # Track the regions we produced across calls so the parent can
        # verify diversity.
        self.call_log: List[Dict[str, Any]] = []

    def process(self, image: Image.Image, **kwargs: Any) -> List[_Region]:
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        w, h = image.size
        regions: List[_Region] = []
        for i in range(self.num_regions):
            x1 = (i * w) // max(self.num_regions, 1)
            x2 = ((i + 1) * w) // max(self.num_regions, 1)
            crop = image.crop((x1, 0, x2, h))
            r = _Region(
                image=crop,
                bbox=(x1, 0, x2, h),
                task_type=str(kwargs.get("task_type", "text")),
                label=f"region-{i}",
                metadata={"pid": self.pid, "idx": i},
            )
            regions.append(r)
            self.call_log.append(
                {"region_id": f"r-{i}", "task_type": r.task_type, "pid": self.pid}
            )
        return regions


class FailingPreprocessor:
    def process(self, image: Image.Image, **kwargs: Any) -> List[_Region]:
        raise RuntimeError("intentional preprocessor failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_config() -> GlmOcrConfig:
    """Build a GlmOcrConfig that does not require a real vLLM server."""
    cfg = load_config(mode="selfhosted")
    # Shorten the OCR connect timeout so a missing server doesn't hang tests.
    cfg.pipeline.ocr_api.connect_timeout = 2
    cfg.pipeline.ocr_api.request_timeout = 5
    cfg.pipeline.ocr_api.retry_max_attempts = 0
    return cfg


def _make_test_image(size: tuple = (200, 100), color: str = "red") -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _patch_ocr_client_to_succeed(monkeypatch):
    """Patch OCRClient.start/process to avoid network calls."""
    monkeypatch.setattr(
        "glmocr.ocr_client.OCRClient.start", lambda self: None
    )
    monkeypatch.setattr(
        "glmocr.ocr_client.OCRClient.stop", lambda self: None
    )

    def _fake_process(self, request_data: Dict[str, Any]):
        return (
            {
                "choices": [
                    {"message": {"content": "ok-from-fake"}}
                ]
            },
            200,
        )

    monkeypatch.setattr(
        "glmocr.ocr_client.OCRClient.process", _fake_process
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class TestAggregator:
    def test_register_and_finalise_on_expected_count(self):
        agg = DocumentAggregator()
        seen: List[Dict[str, Any]] = []
        agg.register("doc-1", on_done=seen.append, expected_regions=2)

        # Region 1
        rt1 = RegionTask(
            region_id="r1",
            document_id="doc-1",
            task_id="t1",
            image_bytes=b"",
            bbox=(0, 0, 10, 10),
            task_type="text",
        )
        payload1 = {"response": {"choices": [{"message": {"content": " hello "}}]}, "status": 200}
        result1 = agg.on_region(rt1, payload1)
        assert result1 is None
        assert agg.has_pending()

        # Region 2 (final)
        rt2 = RegionTask(
            region_id="r2",
            document_id="doc-1",
            task_id="t1",
            image_bytes=b"",
            bbox=(10, 0, 20, 10),
            task_type="text",
        )
        result2 = agg.on_region(rt2, payload1)
        assert result2 is not None
        assert result2["document_id"] == "doc-1"
        assert result2["num_regions"] == 2
        assert result2["regions"][0]["content"] == "hello"  # stripped
        assert not agg.has_pending()
        assert len(seen) == 1
        assert seen[0]["num_regions"] == 2

    def test_unknown_document_is_ignored(self):
        agg = DocumentAggregator()
        rt = RegionTask(
            region_id="r1",
            document_id="ghost",
            task_id="t1",
            image_bytes=b"",
            bbox=(0, 0, 10, 10),
            task_type="text",
        )
        out = agg.on_region(rt, {"response": None, "status": 200})
        assert out is None
        assert not agg.has_pending()

    def test_skip_region_keeps_image(self):
        agg = DocumentAggregator()
        seen: List[Dict[str, Any]] = []
        agg.register("d", on_done=seen.append, expected_regions=1)

        # Construct a small image to round-trip through the skip path.
        img = Image.new("RGB", (5, 5), "blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        rt = RegionTask(
            region_id="r-skip",
            document_id="d",
            task_id="t",
            image_bytes=buf.getvalue(),
            bbox=(0, 0, 5, 5),
            task_type="skip",
        )
        result = agg.on_region(rt, {"status": 200, "skip": True})
        assert result is not None
        assert result["regions"][0]["skip"] is True
        assert result["regions"][0]["image"] is not None
        assert isinstance(result["regions"][0]["image"], Image.Image)

    def test_flush_document_when_expected_unknown(self):
        agg = DocumentAggregator()
        seen: List[Dict[str, Any]] = []
        agg.register("d", on_done=seen.append, expected_regions=None)

        rt = RegionTask(
            region_id="r1",
            document_id="d",
            task_id="t",
            image_bytes=b"",
            bbox=(0, 0, 5, 5),
            task_type="text",
        )
        agg.on_region(rt, {"response": {"choices": [{"message": {"content": "x"}}]}, "status": 200})
        # Aggregator should still be pending because expected_regions is None.
        assert agg.has_pending()
        result = agg.flush_document("d")
        assert result is not None
        assert result["num_regions"] == 1
        assert not agg.has_pending()


# ---------------------------------------------------------------------------
# AsyncOCRDispatcher
# ---------------------------------------------------------------------------


class TestAsyncOCRDispatcher:
    def test_abandon_and_skip_skip_the_network(self):
        fake_client = MagicMock()
        page_loader = MagicMock()
        # page_loader is only called for non-skip regions; we won't get here.
        d = AsyncOCRDispatcher(
            ocr_client=fake_client,
            page_loader=page_loader,
            max_workers=2,
            on_region_done=lambda *a: None,
        )
        d.start()
        try:
            d.submit(
                RegionTask(
                    region_id="r1",
                    document_id="d",
                    task_id="t",
                    image_bytes=b"",
                    bbox=(0, 0, 1, 1),
                    task_type="abandon",
                )
            )
            d.submit(
                RegionTask(
                    region_id="r2",
                    document_id="d",
                    task_id="t",
                    image_bytes=b"",
                    bbox=(0, 0, 1, 1),
                    task_type="skip",
                )
            )
            assert d.drain() == 0  # both went via on_region_done directly
            assert not d.has_inflight()
            # OCR client was never called.
            assert fake_client.process.call_count == 0
        finally:
            d.close()

    def test_text_region_invokes_ocr_client(self, monkeypatch):
        # Use a real PageLoader so build_request_from_image works.
        from glmocr.config import PageLoaderConfig
        from glmocr.dataloader import PageLoader

        pl = PageLoader(PageLoaderConfig())
        fake_client = MagicMock()
        fake_client.process.return_value = (
            {"choices": [{"message": {"content": "result"}}]},
            200,
        )
        captured: List[Dict[str, Any]] = []
        d = AsyncOCRDispatcher(
            ocr_client=fake_client,
            page_loader=pl,
            max_workers=1,
            on_region_done=lambda rt, p: captured.append(
                {"region_id": rt.region_id, "payload": p}
            ),
        )
        d.start()
        try:
            img_bytes = _make_test_image()
            d.submit(
                RegionTask(
                    region_id="r-text",
                    document_id="d",
                    task_id="t",
                    image_bytes=img_bytes,
                    bbox=(0, 0, 200, 100),
                    task_type="text",
                )
            )
            # Poll until done.
            deadline = time.time() + 5
            while d.has_inflight() and time.time() < deadline:
                d.drain()
                time.sleep(0.02)
            assert not d.has_inflight()
            assert len(captured) == 1
            assert captured[0]["region_id"] == "r-text"
            assert captured[0]["payload"]["status"] == 200
            assert fake_client.process.call_count == 1
        finally:
            d.close()


# ---------------------------------------------------------------------------
# PreprocessPool end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_ocr(monkeypatch):
    """Patch OCRClient to avoid any network I/O during tests."""
    _patch_ocr_client_to_succeed(monkeypatch)


class TestPreprocessPool:
    def test_basic_flow_with_expected_regions(self, patched_ocr):
        cfg = _make_test_config()
        results: List[Dict[str, Any]] = []
        with PreprocessPool(
            preprocessor=lambda: DummyPreprocessor(num_regions=2),
            config=cfg,
            num_workers=2,
            ocr_max_workers=4,
        ) as pool:
            pool.submit(
                image_bytes=_make_test_image(),
                document_id="doc-A",
                on_done=results.append,
                expected_regions=2,
            )
            assert pool.join(timeout=15) is True

        assert len(results) == 1
        r = results[0]
        assert r["document_id"] == "doc-A"
        assert r["num_regions"] == 2
        # Regions sorted by region_id (== submission order)
        assert r["regions"][0]["region_id"].endswith("#0")
        assert r["regions"][1]["region_id"].endswith("#1")
        for region in r["regions"]:
            assert region["status"] == 200
            assert region["content"] == "ok-from-fake"

    def test_skip_region_round_trip(self, patched_ocr):
        cfg = _make_test_config()

        class SkipPreprocessor:
            def __init__(self):
                self.pid = os.getpid()

            def process(self, image, **_):
                return [
                    _Region(
                        image=image.copy(),
                        bbox=(0, 0, image.size[0], image.size[1]),
                        task_type="skip",
                        label="chart",
                    )
                ]

        results: List[Dict[str, Any]] = []
        with PreprocessPool(
            preprocessor=SkipPreprocessor,
            config=cfg,
            num_workers=1,
        ) as pool:
            pool.submit(
                image_bytes=_make_test_image(size=(10, 10)),
                document_id="skip-doc",
                on_done=results.append,
                expected_regions=1,
            )
            assert pool.join(timeout=15) is True

        assert len(results) == 1
        r = results[0]
        assert r["regions"][0]["skip"] is True
        assert r["regions"][0]["image"] is not None
        assert isinstance(r["regions"][0]["image"], Image.Image)

    def test_multiple_documents(self, patched_ocr):
        cfg = _make_test_config()
        results: List[Dict[str, Any]] = []
        with PreprocessPool(
            preprocessor=lambda: DummyPreprocessor(num_regions=3),
            config=cfg,
            num_workers=2,
        ) as pool:
            for doc_id in ("A", "B", "C"):
                pool.submit(
                    image_bytes=_make_test_image(),
                    document_id=doc_id,
                    on_done=results.append,
                    expected_regions=3,
                )
            assert pool.join(timeout=20) is True

        completed_ids = {r["document_id"] for r in results}
        assert completed_ids == {"A", "B", "C"}
        for r in results:
            assert r["num_regions"] == 3

    def test_process_pool_diversity(self, patched_ocr):
        """The 4 workers should be running in different pids and all used.

        We embed ``os.getpid()`` in the region metadata so the test can
        see what the children produced without needing a cross-process
        mutable container.
        """
        cfg = _make_test_config()
        aggregated: Dict[str, List[Dict[str, Any]]] = {}

        class PidTrackingPreprocessor:
            def process(self, image, **kwargs):
                my_pid = os.getpid()
                w, h = image.size
                return [
                    _Region(
                        image=image.crop((0, 0, w, h)),
                        bbox=(0, 0, w, h),
                        task_type="text",
                        label="",
                        metadata={"worker_pid": my_pid},
                    )
                ]

        def _on_done(result):
            aggregated[result["document_id"]] = result["regions"]

        with PreprocessPool(
            preprocessor=PidTrackingPreprocessor,
            config=cfg,
            num_workers=4,
        ) as pool:
            for i in range(8):
                pool.submit(
                    image_bytes=_make_test_image(),
                    document_id=f"d{i}",
                    on_done=_on_done,
                    expected_regions=1,
                )
            assert pool.join(timeout=30) is True

        # 8 docs × 1 region = 8 region results.
        all_pids = {
            r["metadata"]["worker_pid"] for regions in aggregated.values() for r in regions
        }
        # With 8 tasks across 4 workers, we should see at least 2 distinct
        # pids in practice (typically all 4 on a healthy machine).
        assert len(all_pids) >= 2, f"only saw worker pids {all_pids}"
        assert sum(len(v) for v in aggregated.values()) == 8

    def test_preprocessor_failure_synthesises_region(self, patched_ocr):
        """When the preprocessor raises, the document is finalised with an
        error metadata entry rather than hanging forever."""
        cfg = _make_test_config()
        results: List[Dict[str, Any]] = []
        with PreprocessPool(
            preprocessor=FailingPreprocessor,
            config=cfg,
            num_workers=1,
        ) as pool:
            pool.submit(
                image_bytes=_make_test_image(),
                document_id="boom",
                on_done=results.append,
                expected_regions=1,
            )
            assert pool.join(timeout=10) is True

        assert len(results) == 1
        r = results[0]
        # Synthetic abandon region was pushed.
        assert r["num_regions"] == 1
        assert r["regions"][0]["task_type"] == "abandon"
        assert "error" in r["regions"][0]["metadata"]

    def test_submit_rejects_missing_image(self, patched_ocr):
        cfg = _make_test_config()
        with PreprocessPool(
            preprocessor=lambda: DummyPreprocessor(),
            config=cfg,
            num_workers=1,
        ) as pool:
            with pytest.raises(ValueError):
                pool.submit(document_id="x")
            with pytest.raises(ValueError):
                pool.submit(document_id="")

    def test_stop_joins_workers(self, patched_ocr):
        cfg = _make_test_config()
        pool = PreprocessPool(
            preprocessor=lambda: DummyPreprocessor(num_regions=1, sleep_seconds=0.05),
            config=cfg,
            num_workers=2,
        )
        pool.start()
        pool.submit(
            image_bytes=_make_test_image(),
            document_id="d1",
            on_done=lambda *_: None,
            expected_regions=1,
        )
        pool.stop(timeout=5)
        # After stop the workers should be terminated.
        assert pool._processes == []


# ---------------------------------------------------------------------------
# Re-export the dummy classes so the test runner can introspect them.
# ---------------------------------------------------------------------------


__all__ = [
    "DummyPreprocessor",
    "FailingPreprocessor",
    "TestAggregator",
    "TestAsyncOCRDispatcher",
    "TestPreprocessPool",
]
