"""Manual smoke test for PreprocessPool.

Run with: ``python smoke_test.py`` from the repo root.

This file is **not** picked up by pytest (it doesn't start with
``test_``) so it won't interfere with the unit tests.
"""

from __future__ import annotations

import io
import sys
import time
from unittest.mock import patch

# Make sure fork is used so this top-level class is inherited by workers.
import multiprocessing

multiprocessing.set_start_method("fork", force=True)

from PIL import Image

from glmocr.config import load_config
from glmocr.preprocess_pool import PreprocessPool, Region


# ---- Top-level preprocessor (must be at module scope to be picklable) ----
class SmokePreprocessor:
    """Split each image in half and emit two text regions."""

    def process(self, image, **_):
        w, h = image.size
        return [
            Region(
                image=image.crop((0, 0, w // 2, h)),
                bbox=(0, 0, w // 2, h),
                task_type="text",
                label="L",
                metadata={"half": "left"},
            ),
            Region(
                image=image.crop((w // 2, 0, w, h)),
                bbox=(w // 2, 0, w, h),
                task_type="text",
                label="R",
                metadata={"half": "right"},
            ),
        ]


def main() -> int:
    cfg = load_config(mode="selfhosted")
    cfg.pipeline.ocr_api.connect_timeout = 1
    cfg.pipeline.ocr_api.retry_max_attempts = 0

    img = Image.new("RGB", (200, 50), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    results = []
    seen: list[float] = []

    def on_done(result):
        seen.append(time.time())
        results.append(result)

    with patch(
        "glmocr.ocr_client.OCRClient.start", lambda self: None
    ), patch("glmocr.ocr_client.OCRClient.stop", lambda self: None), patch(
        "glmocr.ocr_client.OCRClient.process",
        lambda self, req: (
            {"choices": [{"message": {"content": "demo"}}]},
            200,
        ),
    ):
        with PreprocessPool(SmokePreprocessor, cfg, num_workers=2) as pool:
            for doc_id in ("doc-1", "doc-2"):
                pool.submit(
                    image_bytes=img_bytes,
                    document_id=doc_id,
                    on_done=on_done,
                    expected_regions=2,
                )
            ok = pool.join(timeout=15)
            if not ok:
                print("TIMEOUT")
                return 1

    print(f"Got {len(results)} results, {len(seen)} callbacks fired")
    for r in results:
        print(
            f"  - {r['document_id']}: {r['num_regions']} regions, "
            f"first content={r['regions'][0]['content']!r}, "
            f"half={r['regions'][0]['metadata']['half']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
