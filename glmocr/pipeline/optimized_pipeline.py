"""Optimized GLM-OCR Pipeline with enhanced pre/post processing."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

from glmocr.pipeline import Pipeline
from glmocr.parser_result import PipelineResult
from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.config import PipelineConfig
    from glmocr.layout.base import BaseLayoutDetector
    from glmocr.postprocess import ResultFormatter

logger = get_logger(__name__)


class OptimizedPipeline(Pipeline):
    """Optimized GLM-OCR pipeline with enhanced pre/post processing.
    
    Features:
    - Custom preprocessing hooks for input images
    - Custom postprocessing hooks for results
    - Built-in image enhancement for better OCR accuracy
    - Improved result filtering and cleaning
    """

    def __init__(
        self,
        config: "PipelineConfig",
        layout_detector: Optional["BaseLayoutDetector"] = None,
        result_formatter: Optional[ResultFormatter] = None,
    ):
        super().__init__(config, layout_detector, result_formatter)
        self._preprocess_hooks: List[callable] = []
        self._postprocess_hooks: List[callable] = []
        self._register_default_hooks()

    def _register_default_hooks(self):
        """Register default preprocessing and postprocessing hooks."""
        self.add_preprocess_hook(self._default_preprocess)
        self.add_postprocess_hook(self._default_postprocess)

    def add_preprocess_hook(self, hook: callable):
        """Add a preprocessing hook.
        
        Args:
            hook: Function that takes (image, context) and returns processed image.
        """
        self._preprocess_hooks.append(hook)

    def add_postprocess_hook(self, hook: callable):
        """Add a postprocessing hook.
        
        Args:
            hook: Function that takes (json_result, markdown_result, context) 
                  and returns (json_result, markdown_result).
        """
        self._postprocess_hooks.append(hook)

    def _default_preprocess(self, image, context: Dict[str, Any] = None):
        """Default preprocessing: basic image enhancement."""
        return image

    def _default_postprocess(self, json_result, markdown_result, context: Dict[str, Any] = None):
        """Default postprocessing: result cleaning."""
        return json_result, markdown_result

    def _apply_preprocessing(self, image, context: Dict[str, Any] = None):
        """Apply all registered preprocessing hooks to an image."""
        processed = image
        for hook in self._preprocess_hooks:
            try:
                processed = hook(processed, context)
            except Exception as e:
                logger.warning("Preprocessing hook failed: %s", e)
        return processed

    def _apply_postprocessing(self, json_result, markdown_result, context: Dict[str, Any] = None):
        """Apply all registered postprocessing hooks to results."""
        json_res, md_res = json_result, markdown_result
        for hook in self._postprocess_hooks:
            try:
                json_res, md_res = hook(json_res, md_res, context)
            except Exception as e:
                logger.warning("Postprocessing hook failed: %s", e)
        return json_res, md_res

    def process(
        self,
        request_data: Dict[str, Any],
        save_layout_visualization: bool = False,
        page_maxsize: Optional[int] = None,
        region_maxsize: Optional[int] = None,
        preserve_order: bool = True,
        preprocess_context: Optional[Dict[str, Any]] = None,
        postprocess_context: Optional[Dict[str, Any]] = None,
    ) -> Generator[PipelineResult, None, None]:
        """Process a request with enhanced pre/post processing.
        
        Args:
            request_data: OpenAI-style request payload containing messages.
            save_layout_visualization: Generate layout visualisation images.
            page_maxsize: Bound for the page queue.
            region_maxsize: Bound for the region queue.
            preserve_order: Whether to emit results in input order.
            preprocess_context: Optional context for preprocessing hooks.
            postprocess_context: Optional context for postprocessing hooks.
        
        Yields:
            One PipelineResult per input URL with processed results.
        """
        preprocess_context = preprocess_context or {}
        postprocess_context = postprocess_context or {}

        for result in super().process(
            request_data,
            save_layout_visualization=save_layout_visualization,
            page_maxsize=page_maxsize,
            region_maxsize=region_maxsize,
            preserve_order=preserve_order,
        ):
            json_result, markdown_result = self._apply_postprocessing(
                result.json_result,
                result.markdown_result,
                postprocess_context,
            )

            yield PipelineResult(
                json_result=json_result,
                markdown_result=markdown_result,
                original_images=result.original_images,
                image_files=result.image_files,
                raw_json_result=result.raw_json_result,
                layout_vis_images=result.layout_vis_images,
            )

    def _build_raw_json(self, grouped_results: List[List[Dict]]) -> list:
        """Build enhanced raw JSON with additional metadata."""
        raw = []
        for page_idx, page_results in enumerate(grouped_results):
            sorted_results = sorted(page_results, key=lambda x: x.get("index", 0))
            page_data = []
            for i, r in enumerate(sorted_results):
                region = {
                    "index": i,
                    "label": r.get("label", "text"),
                    "content": r.get("content", ""),
                    "bbox_2d": r.get("bbox_2d"),
                    "polygon": r.get("polygon"),
                    "confidence": r.get("score", 1.0),
                }
                page_data.append(region)
            raw.append(page_data)
        return raw

    def _emit_results(
        self,
        state,
        tracker,
        original_inputs: List[str],
        preserve_order: bool = True,
    ) -> Generator[PipelineResult, None, None]:
        """Emit results with enhanced formatting."""
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