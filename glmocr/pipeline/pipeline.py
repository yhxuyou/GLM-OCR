"""GLM-OCR Pipeline

Three-stage async document parsing pipeline.  ``process()`` yields one
``PipelineResult`` per input unit (one image or one PDF).

Stages (all always enabled):
  1. PageLoader   — load images / PDF pages
  2. LayoutDetector — detect regions per page
  3. OCRClient    — recognise each region via VLM

Extension points:
  * Pass a custom ``layout_detector`` or ``result_formatter`` to the constructor.
  * Subclass ``Pipeline`` and override ``process()``.
"""

from __future__ import annotations

import asyncio
import time
import threading
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

from glmocr.aggregator import RegionAggregator
from glmocr.dataloader import PageLoader
from glmocr.ocr_client import OCRClient
from glmocr.parser_result import PipelineResult
from glmocr.postprocess import ResultFormatter
from glmocr.utils.logging import get_logger

from glmocr.pipeline._common import (
    extract_image_sources,
    extract_ocr_content,
    make_original_inputs,
)
from glmocr.pipeline._state import PipelineState
from glmocr.pipeline._workers import (
    data_loading_worker,
    layout_worker,
    recognition_worker,
)
from glmocr.pipeline._unit_tracker import UnitTracker

if TYPE_CHECKING:
    from glmocr.config import PipelineConfig
    from glmocr.layout.base import BaseLayoutDetector

logger = get_logger(__name__)


class Pipeline:
    """GLM-OCR pipeline.

    Processing flow:
      1. PageLoader:      load images / PDF into pages
      2. LayoutDetector:  detect regions
      3. OCRClient:       call OCR service
      4. ResultFormatter: format outputs

    Args:
        config: PipelineConfig instance.
        layout_detector: Custom layout detector (optional).
        result_formatter: Custom result formatter (optional).

    Example::

        from glmocr.config import load_config

        cfg = load_config()
        pipeline = Pipeline(cfg.pipeline)
        for result in pipeline.process(request_data):
            result.save(output_dir="./output")
    """

    def __init__(
        self,
        config: "PipelineConfig",
        layout_detector: Optional["BaseLayoutDetector"] = None,
        result_formatter: Optional[ResultFormatter] = None,
    ):
        self.config = config
        self.page_loader = PageLoader(config.page_loader)
        self.ocr_client = OCRClient(config.ocr_api)
        self.result_formatter = (
            result_formatter
            if result_formatter is not None
            else ResultFormatter(config.result_formatter)
        )

        if layout_detector is not None:
            self.layout_detector = layout_detector
        else:
            from glmocr.layout import PPDocLayoutDetector

            if PPDocLayoutDetector is None:
                from glmocr.layout import _raise_layout_import_error

                _raise_layout_import_error()

            self.layout_detector = PPDocLayoutDetector(config.layout)

        self.max_workers = config.max_workers
        self._page_maxsize = getattr(config, "page_maxsize", 100)
        self._region_maxsize = getattr(config, "region_maxsize", 2000)
        self._current_state: Optional[PipelineState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        request_data: Dict[str, Any],
        save_layout_visualization: bool = False,
        page_maxsize: Optional[int] = None,
        region_maxsize: Optional[int] = None,
        preserve_order: bool = True,
    ) -> Generator[PipelineResult, None, None]:
        """Process a request; yield one ``PipelineResult`` per input unit.

        Uses three threads (load → layout → recognition) with bounded queues
        for back-pressure.

        Args:
            request_data: OpenAI-style request payload containing messages.
            save_layout_visualization: Generate layout visualisation images.
            page_maxsize: Bound for the page queue.
            region_maxsize: Bound for the region queue.
            preserve_order: Whether to emit results in input order.

        Yields:
            One ``PipelineResult`` per input URL (image or PDF).
        """
        image_sources = extract_image_sources(request_data)

        if not image_sources:
            yield self._process_passthrough(request_data)
            return

        num_units = len(image_sources)
        original_inputs = make_original_inputs(image_sources)

        state = PipelineState(
            page_maxsize=page_maxsize or self._page_maxsize,
            region_maxsize=region_maxsize or self._region_maxsize,
        )
        self._current_state = state

        tracker = UnitTracker(num_units)
        state.set_tracker(tracker)

        t1 = threading.Thread(
            target=data_loading_worker,
            args=(state, self.page_loader, image_sources),
            daemon=True,
        )
        t2 = threading.Thread(
            target=layout_worker,
            args=(
                state,
                self.layout_detector,
                save_layout_visualization,
                self.config.layout.use_polygon,
            ),
            daemon=True,
        )
        t3 = threading.Thread(
            target=recognition_worker,
            args=(state, self.page_loader, self.ocr_client, self.max_workers),
            daemon=True,
        )

        t1.start()
        t2.start()
        t3.start()

        t_watchdog = threading.Thread(
            target=self._health_watchdog,
            args=(state,),
            daemon=True,
        )
        t_watchdog.start()

        try:
            yield from self._emit_results(
                state, tracker, original_inputs, preserve_order=preserve_order
            )
        finally:
            state.request_shutdown()
            t1.join(timeout=10)
            t2.join(timeout=10)
            t3.join(timeout=10)
            t_watchdog.join(timeout=5)
            self._current_state = None

        state.raise_if_exceptions()

    # ------------------------------------------------------------------
    # Async Pipeline Support
    # ------------------------------------------------------------------

    async def process_async(
        self,
        request_data: Dict[str, Any],
        doc_id: str,
        aggregator: RegionAggregator,
        save_layout_visualization: bool = False,
    ) -> str:
        """Process a request asynchronously and return immediately.

        This method performs layout detection synchronously, then submits all
        regions to vLLM asynchronously without waiting for results. The
        ``doc_id`` is returned immediately, and results can be retrieved later
        via the ``aggregator``.

        Args:
            request_data: OpenAI-style request payload containing messages.
            doc_id: Unique identifier for this document.
            aggregator: RegionAggregator instance to track progress.
            save_layout_visualization: Generate layout visualisation images.

        Returns:
            The ``doc_id`` for tracking progress.
        """
        image_sources = extract_image_sources(request_data)

        if not image_sources:
            # No images - process synchronously and store result
            result = self._process_passthrough(request_data)
            await aggregator.register_document(doc_id, total_regions=1)
            await aggregator.on_region_complete(
                doc_id,
                "passthrough",
                {
                    "json_result": result.json_result,
                    "markdown_result": result.markdown_result,
                },
            )
            return doc_id

        # Load all pages
        pages = []
        for page, _ in self.page_loader.iter_pages_with_unit_indices(image_sources):
            pages.append(page)

        if not pages:
            await aggregator.register_document(doc_id, total_regions=1)
            await aggregator.on_region_complete(
                doc_id,
                "empty",
                {"json_result": None, "markdown_result": ""},
            )
            return doc_id

        # Run layout detection on all pages
        all_regions = []
        for page_idx, page in enumerate(pages):
            try:
                layout_results, vis_images = self.layout_detector.process(
                    [page],
                    save_visualization=save_layout_visualization,
                    global_start_idx=page_idx,
                    use_polygon=self.config.layout.use_polygon,
                )
                if layout_results:
                    for region in layout_results[0]:
                        all_regions.append({
                            "page_idx": page_idx,
                            "page": page,
                            "region": region,
                        })
            except Exception as e:
                logger.warning(
                    "Layout detection failed for page %d, skipping: %s",
                    page_idx,
                    e,
                )

        # Register document with aggregator
        total_regions = len(all_regions)
        if total_regions == 0:
            total_regions = 1  # At least one region to avoid ValueError

        await aggregator.register_document(doc_id, total_regions=total_regions)

        if not all_regions:
            await aggregator.on_region_complete(
                doc_id,
                "no_regions",
                {"json_result": None, "markdown_result": ""},
            )
            return doc_id

        # Submit all regions asynchronously
        tasks = []
        for idx, region_info in enumerate(all_regions):
            region_id = f"region_{idx}"
            task = asyncio.create_task(
                self._submit_region_async(doc_id, region_id, region_info, aggregator)
            )
            tasks.append(task)

        # Wait for all submissions to complete (but not for OCR results)
        # The tasks will continue running in the background
        logger.info(
            "Submitted %d regions for document %s asynchronously",
            len(tasks),
            doc_id,
        )

        return doc_id

    async def _submit_region_async(
        self,
        doc_id: str,
        region_id: str,
        region_info: Dict[str, Any],
        aggregator: RegionAggregator,
    ) -> None:
        """Submit a single region to vLLM asynchronously.

        Args:
            doc_id: Document identifier.
            region_id: Region identifier within the document.
            region_info: Dictionary containing page, region, and cropped image info.
            aggregator: RegionAggregator to notify on completion.
        """
        page_idx = region_info["page_idx"]
        page = region_info["page"]
        region = region_info["region"]

        try:
            # Handle skip task types
            if region.get("task_type") == "skip":
                region["content"] = None
                result = {
                    "page_idx": page_idx,
                    "region": region,
                    "content": None,
                }
                await aggregator.on_region_complete(doc_id, region_id, result)
                return

            # Crop the image for this region
            from glmocr.utils.image_utils import crop_image_region

            try:
                polygon = region.get("polygon") if self.config.layout.use_polygon else None
                cropped_image = crop_image_region(
                    page, region["bbox_2d"], polygon
                )
            except Exception as e:
                logger.warning(
                    "Failed to crop region on page %d (bbox=%s), skipping: %s",
                    page_idx,
                    region.get("bbox_2d"),
                    e,
                )
                region["content"] = ""
                result = {
                    "page_idx": page_idx,
                    "region": region,
                    "content": "",
                }
                await aggregator.on_region_complete(doc_id, region_id, result)
                return

            # Build request from cropped image
            request_data = self.page_loader.build_request_from_image(
                cropped_image,
                region.get("task_type", "text"),
            )

            # Submit OCR request in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response, status_code = await loop.run_in_executor(
                None, self.ocr_client.process, request_data
            )

            # Process response
            if status_code == 200:
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                region["content"] = content.strip() if content else ""
            else:
                logger.warning(
                    "Recognition failed for page %d region %s: HTTP %s",
                    page_idx,
                    region_id,
                    status_code,
                )
                region["content"] = None

            result = {
                "page_idx": page_idx,
                "region": region,
                "content": region.get("content"),
            }
            await aggregator.on_region_complete(doc_id, region_id, result)

        except Exception as e:
            logger.error(
                "Async region submission failed for doc=%s region=%s: %s",
                doc_id,
                region_id,
                e,
            )
            # Mark region as failed
            result = {
                "page_idx": page_idx,
                "region": region,
                "content": None,
                "error": str(e),
            }
            await aggregator.on_region_complete(doc_id, region_id, result)

    def get_queue_stats(self) -> Optional[Dict[str, int]]:
        """Return current queue sizes, or ``None`` if no processing is active."""
        state = self._current_state
        if state is None:
            return None
        return {
            "page_queue_size": state.page_queue.qsize(),
            "page_queue_maxsize": state.page_queue.maxsize,
            "region_queue_size": state.region_queue.qsize(),
            "region_queue_maxsize": state.region_queue.maxsize,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the pipeline (layout detector + OCR client)."""
        logger.info("Starting Pipeline...")
        self.layout_detector.start()
        self.ocr_client.start()
        logger.info("Pipeline started!")

    def stop(self):
        """Stop the pipeline."""
        logger.info("Stopping Pipeline...")
        self.ocr_client.stop()
        self.layout_detector.stop()
        logger.info("Pipeline stopped!")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ------------------------------------------------------------------
    # Health watchdog
    # ------------------------------------------------------------------

    def _health_watchdog(
        self,
        state: PipelineState,
        check_interval: float = 5.0,
    ) -> None:
        """Daemon thread that monitors OCR service liveness.

        Periodically probes the API port via socket.  On the first
        failure the pipeline is shut down immediately so that workers
        stop instead of accumulating failed requests.
        """
        while not state.is_shutdown:
            state._shutdown_event.wait(check_interval)
            if state.is_shutdown:
                break

            if not self.ocr_client.is_alive():
                error = RuntimeError(
                    f"OCR service at {self.ocr_client.api_host}:{self.ocr_client.api_port} "
                    f"is no longer available"
                )
                logger.error("%s", error)
                state.record_exception("HealthWatchdog", error)
                break

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_raw_json(grouped_results: List[List[Dict]]) -> list:
        """Build a raw JSON snapshot from grouped recognition results.

        Same structure as the final JSON (list of pages, each a list of region
        dicts) but with the original model output before any post-processing.
        """
        raw = []
        for page_results in grouped_results:
            sorted_results = sorted(page_results, key=lambda x: x.get("index", 0))
            raw.append(
                [
                    {
                        "index": i,
                        "label": r.get("label", "text"),
                        "content": r.get("content", ""),
                        "bbox_2d": r.get("bbox_2d"),
                        "polygon": r.get("polygon"),
                    }
                    for i, r in enumerate(sorted_results)
                ]
            )
        return raw

    def _process_passthrough(
        self,
        request_data: Dict[str, Any],
    ) -> PipelineResult:
        """No image URLs — forward the request directly to the OCR API."""
        request_data = self.page_loader.build_request(request_data)
        response, status_code = self.ocr_client.process(request_data)
        if status_code != 200:
            raise Exception(
                f"OCR request failed: {response}, status_code: {status_code}"
            )
        content = extract_ocr_content(response)
        json_result, markdown_result = self.result_formatter.format_ocr_result(content)
        return PipelineResult(
            json_result=json_result,
            markdown_result=markdown_result,
            original_images=[],
        )

    def _emit_results(
        self,
        state: PipelineState,
        tracker: UnitTracker,
        original_inputs: List[str],
        preserve_order: bool = True,
    ) -> Generator[PipelineResult, None, None]:
        """Wait for units to complete and yield formatted results.

        When ``preserve_order`` is True, units may complete in arbitrary order
        but are buffered and yielded sequentially (unit 0, 1, 2, ...).
        When ``preserve_order`` is False, each ready unit is yielded immediately.

        ``None`` from the ready queue signals a pipeline error (shutdown).
        """
        pending: Dict[int, PipelineResult] = {}
        built: set = set()
        next_to_emit = 0
        num_units = tracker.num_units

        while (
            (next_to_emit < num_units) if preserve_order else (len(built) < num_units)
        ):
            if preserve_order:
                while next_to_emit in pending:
                    yield pending.pop(next_to_emit)
                    next_to_emit += 1
                if next_to_emit >= num_units:
                    break

            u = tracker.wait_next_ready_unit()
            if u is None:
                break
            if u in built:
                continue

            region_count = tracker.unit_region_count[u]
            if region_count is None:
                tracker._ready_queue.put(u)
                time.sleep(0.05)
                continue

            page_indices = tracker.unit_image_indices[u]
            grouped = state.get_grouped_results(page_indices)

            total = sum(len(g) for g in grouped)
            if total < region_count:
                tracker._ready_queue.put(u)
                time.sleep(0.05)
                continue

            cropped_images = state.collect_cropped_images_for_unit(page_indices)
            raw_json = self._build_raw_json(grouped)
            json_u, md_u, image_files = self.result_formatter.process(
                grouped,
                cropped_images=cropped_images or None,
            )

            vis_images = {}
            for pi in page_indices:
                img = state.layout_vis_images.pop(pi, None)
                if img is not None:
                    vis_images[pi] = img

            state.release_unit_data(page_indices)

            result = PipelineResult(
                json_result=json_u,
                markdown_result=md_u,
                original_images=[original_inputs[u]],
                image_files=image_files or None,
                raw_json_result=raw_json,
                layout_vis_images=vis_images or None,
            )
            built.add(u)
            if preserve_order:
                pending[u] = result
            else:
                yield result
