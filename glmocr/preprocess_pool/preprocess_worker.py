"""Child-process worker entry point.

Each worker process instantiates the user's :class:`DocumentPreprocessor`
once at start-up, then loops forever reading :class:`PreprocessTask`
messages from the input queue, running the four-stage preprocessing, and
pushing the resulting :class:`RegionTask` messages back through the output
queue.

Logging is configured independently inside each worker so that log lines
are tagged with the worker's pid, which is essential when debugging
issues that only appear in one of the four processes.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from typing import Callable

from PIL import Image

from glmocr.preprocess_pool.protocols import DocumentPreprocessor
from glmocr.preprocess_pool.tasks import (
    PreprocessTask,
    RegionError,
    RegionTask,
    WORKER_STOP_SENTINEL,
)
from glmocr.utils.logging import configure_logging, get_logger


# Configure logging early so that the "Preprocess worker ready" line carries
# the worker pid prefix.  We don't propagate to the root handler because the
# parent process owns the root logger.
configure_logging(level=os.environ.get("GLMOCR_LOG_LEVEL", "INFO"))
logger = get_logger(__name__)


def _encode_region_image(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes for cross-process transfer."""
    buf = io.BytesIO()
    # PNG is lossless; for crops this is fine.  Users worried about
    # throughput can override this method via a custom worker.
    if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
        image = image.convert("RGB")
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def preprocess_worker(
    input_q,
    output_q,
    preprocessor_factory: Callable[[], DocumentPreprocessor],
) -> None:
    """Worker main loop.

    Args:
        input_q: ``multiprocessing.Queue`` carrying :class:`PreprocessTask`.
        output_q: ``multiprocessing.Queue`` receiving :class:`RegionTask`
            and :class:`RegionError`.
        preprocessor_factory: Zero-arg callable that returns a fresh
            :class:`DocumentPreprocessor` instance.  Called once per worker.
    """
    try:
        preprocessor = preprocessor_factory()
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to instantiate preprocessor: %s", e)
        # Push a fatal error so the parent can react.
        output_q.put(
            RegionError(
                region_id="<worker-init>",
                document_id="<worker-init>",
                task_id="<worker-init>",
                error=f"preprocessor init failed: {e}",
            )
        )
        return

    logger.info(
        "Preprocess worker ready (pid=%d, preprocessor=%s)",
        os.getpid(),
        type(preprocessor).__name__,
    )

    while True:
        try:
            item = input_q.get()
        except (EOFError, KeyboardInterrupt):  # pragma: no cover
            break
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("input_q.get() failed: %s", e)
            continue

        if item is WORKER_STOP_SENTINEL:
            logger.debug("Worker pid=%d received stop sentinel", os.getpid())
            break

        if not isinstance(item, PreprocessTask):
            logger.warning(
                "Worker pid=%d received unexpected message: %r", os.getpid(), item
            )
            continue

        try:
            image = Image.open(io.BytesIO(item.image_bytes))
            image.load()  # force decode so we don't hold a lazy file handle
        except Exception as e:
            logger.exception("Failed to decode image for task %s", item.task_id)
            output_q.put(
                RegionError(
                    region_id=item.task_id,
                    document_id=item.document_id,
                    task_id=item.task_id,
                    error=f"image decode failed: {e}",
                )
            )
            continue

        try:
            regions = preprocessor.process(image, **item.preprocess_kwargs)
        except Exception as e:
            logger.exception("Preprocess failed for task %s", item.task_id)
            output_q.put(
                RegionError(
                    region_id=item.task_id,
                    document_id=item.document_id,
                    task_id=item.task_id,
                    error=f"preprocess: {e}",
                )
            )
            continue

        for idx, region in enumerate(regions):
            try:
                encoded = _encode_region_image(region.image)
            except Exception as e:
                logger.warning(
                    "Failed to encode region %d of task %s, skipping: %s",
                    idx,
                    item.task_id,
                    e,
                )
                continue

            output_q.put(
                RegionTask(
                    region_id=f"{item.task_id}#{idx}",
                    document_id=item.document_id,
                    task_id=item.task_id,
                    image_bytes=encoded,
                    bbox=tuple(region.bbox) if region.bbox else (0, 0, 0, 0),
                    task_type=region.task_type,
                    label=region.label,
                    polygon=region.polygon,
                    on_done=None,
                    metadata=region.metadata,
                )
            )

    logger.debug("Worker pid=%d exiting", os.getpid())


__all__ = ["preprocess_worker"]
