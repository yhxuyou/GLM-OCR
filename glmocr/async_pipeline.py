"""Asynchronous GLM-OCR Pipeline

Fully async pipeline implementation using AsyncOCRClient for non-blocking
OCR requests. Layout detection still runs in background threads (CPU-intensive),
but OCR submissions are fully asynchronous with semaphore-based concurrency control.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

from glmocr.aggregator import RegionAggregator
from glmocr.dataloader import PageLoader
from glmocr.async_ocr_client import AsyncOCRClient
from glmocr.parser_result import PipelineResult
from glmocr.postprocess import ResultFormatter
from glmocr.utils.logging import get_logger

from glmocr.pipeline._common import (
    extract_image_sources,
    extract_ocr_content,
    make_original_inputs,
)

if TYPE_CHECKING:
    from glmocr.config import PipelineConfig
    from glmocr.layout.base import BaseLayoutDetector

logger = get_logger(__name__)


class AsyncPipeline:
    """Fully asynchronous GLM-OCR pipeline.

    Processing flow:
      1. PageLoader:      load images / PDF into pages (background thread)
      2. LayoutDetector:  detect regions (background thread, CPU-intensive)
      3. AsyncOCRClient:  call OCR service asynchronously
      4. ResultFormatter: format outputs

    Args:
        config: PipelineConfig instance.
        layout_detector: Custom layout detector (optional).
        result_formatter: Custom result formatter (optional).

    Example::

        from glmocr.config import load_config
        from glmocr.async_pipeline import AsyncPipeline

        cfg = load_config()
        pipeline = AsyncPipeline(cfg.pipeline)
        await pipeline.start()
        doc_id = await pipeline.process_async(request_data, doc_id, aggregator)
        # Later: retrieve results via aggregator
        await pipeline.stop()
    """

    def __init__(
        self,
        config: "PipelineConfig",
        layout_detector: Optional["BaseLayoutDetector"] = None,
        result_formatter: Optional[ResultFormatter] = None,
    ):
        self.config = config
        self.page_loader = PageLoader(config.page_loader)
        self.async_ocr_client = AsyncOCRClient(config.ocr_api)
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

        # Concurrency control for region submissions
        max_concurrent_regions = getattr(config, "max_concurrent_regions", 100)
        self._region_semaphore = asyncio.Semaphore(max_concurrent_regions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the pipeline (layout detector + async OCR client)."""
        logger.info("Starting AsyncPipeline...")
        self.layout_detector.start()
        await self.async_ocr_client.start()
        logger.info("AsyncPipeline started!")

    async def stop(self) -> None:
        """Stop the pipeline."""
        logger.info("Stopping AsyncPipeline...")
        await self.async_ocr_client.stop()
        self.layout_detector.stop()
        logger.info("AsyncPipeline stopped!")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def process_async(
        self,
        request_data: Dict[str, Any],
        doc_id: str,
        aggregator: RegionAggregator,
        save_layout_visualization: bool = False,
    ) -> str:
        """Process a request asynchronously and return immediately.

        This method performs layout detection in a background thread, then submits
        all regions to vLLM asynchronously without waiting for results. The
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
            result = await self._process_passthrough_async(request_data)
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

        # Load all pages in background thread (I/O bound)
        loop = asyncio.get_event_loop()
        pages = await loop.run_in_executor(
            None,
            self._load_pages,
            image_sources,
        )

        if not pages:
            await aggregator.register_document(doc_id, total_regions=1)
            await aggregator.on_region_complete(
                doc_id,
                "empty",
                {"json_result": None, "markdown_result": ""},
            )
            return doc_id

        # Run layout detection in background thread (CPU-intensive)
        all_regions = await loop.run_in_executor(
            None,
            self._run_layout_detection,
            pages,
            save_layout_visualization,
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

        # Submit all regions asynchronously with semaphore control
        tasks = []
        for idx, region_info in enumerate(all_regions):
            region_id = f"region_{idx}"
            task = asyncio.create_task(
                self._submit_region_async(doc_id, region_id, region_info, aggregator)
            )
            tasks.append(task)

        # Fire-and-forget: don't wait for tasks to complete
        # They will continue running in the background
        logger.info(
            "Submitted %d regions for document %s asynchronously",
            len(tasks),
            doc_id,
        )

        return doc_id

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pages(self, image_sources: list) -> list:
        """Load all pages from image sources (runs in executor)."""
        pages = []
        for page, _ in self.page_loader.iter_pages_with_unit_indices(image_sources):
            pages.append(page)
        return pages

    def _run_layout_detection(
        self,
        pages: list,
        save_layout_visualization: bool,
    ) -> list:
        """Run layout detection on all pages (runs in executor)."""
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
        return all_regions

    async def _submit_region_async(
        self,
        doc_id: str,
        region_id: str,
        region_info: Dict[str, Any],
        aggregator: RegionAggregator,
    ) -> None:
        """Submit a single region to vLLM asynchronously.

        Uses semaphore to control concurrency.

        Args:
            doc_id: Document identifier.
            region_id: Region identifier within the document.
            region_info: Dictionary containing page, region, and cropped image info.
            aggregator: RegionAggregator to notify on completion.
        """
        async with self._region_semaphore:
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

                # Submit OCR request asynchronously
                response, status_code = await self.async_ocr_client.async_process(
                    request_data
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
                # Mark region as failing
                result = {
                    "page_idx": page_idx,
                    "region": region,
                    "content": None,
                    "error": str(e),
                }
                await aggregator.on_region_complete(doc_id, region_id, result)

    async def _process_passthrough_async(
        self,
        request_data: Dict[str, Any],
    ) -> PipelineResult:
        """No image URLs — forward the request directly to the OCR API asynchronously."""
        request_data = self.page_loader.build_request(request_data)
        response, status_code = await self.async_ocr_client.async_process(request_data)
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
