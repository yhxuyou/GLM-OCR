"""Medical OCR Pipeline - Optimized pipeline for medical documents.

Inherits from glmocr's Pipeline and adds medical-specific pre and post processing.
"""

from typing import Dict, Any, List, Generator, Optional, Callable, Tuple
import re
import json

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)

# 可选依赖
try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

# 导入基础 Pipeline 类
try:
    from glmocr.pipeline import Pipeline
    from glmocr.parser_result import PipelineResult
    BASE_PIPELINE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Base Pipeline not available: {e}")
    Pipeline = object
    PipelineResult = object
    BASE_PIPELINE_AVAILABLE = False


# 只有在基础 Pipeline 可用时才定义我们的类
if BASE_PIPELINE_AVAILABLE and Pipeline:
    class MedicalOcrPipeline(Pipeline):
        """Specialized pipeline for medical OCR.

        Features:
        - Medical document preprocessing hooks
        - Medical terminology post-processing
        - Specialized result formatting for medical records
        - Custom hook system for extendability
        """

        # Common medical terminology patterns for post-processing
        MEDICAL_PATTERNS = {
            # Common units
            "temperature": r"\b(\d+(?:\.\d+)?)\s*(°?C|°?F)\b",
            "weight": r"\b(\d+(?:\.\d+)?)\s*(kg|g|lbs?)\b",
            "height": r"\b(\d+(?:\.\d+)?)\s*(cm|m|ft|in)\b",
            "blood_pressure": r"\b(\d{2,3})/(\d{2,3})\s*(mmHg)?\b",
            "date": r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b",
            "time": r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b",
            "patient_id": r"\b(?:ID|Patient\s*ID|MRN):?\s*([A-Z0-9-]+)\b",
        }

        def __init__(
            self,
            config,
            layout_detector=None,
            result_formatter=None,
        ):
            super().__init__(config, layout_detector, result_formatter)

            # Initialize hook lists
            self.preprocess_hooks: List[Callable] = []
            self.postprocess_hooks: List[Callable] = []

            # Register default hooks
            self._register_default_hooks()

        def _register_default_hooks(self):
            """Register default pre and post processing hooks."""
            # Default preprocessing
            self.add_preprocess_hook(self._default_preprocess)

            # Default postprocessing
            self.add_postprocess_hook(self._medical_terminology_normalization)
            self.add_postprocess_hook(self._clean_medical_text)

        def add_preprocess_hook(self, hook: Callable):
            """Add a custom preprocessing hook."""
            self.preprocess_hooks.append(hook)

        def add_postprocess_hook(self, hook: Callable):
            """Add a custom postprocessing hook."""
            self.postprocess_hooks.append(hook)

        def _default_preprocess(self, image, context: Dict[str, Any] = None):
            """Default preprocessing hook."""
            return image

        def _medical_terminology_normalization(
            self,
            json_result: str,
            markdown_result: str,
            context: Dict[str, Any] = None
        ) -> Tuple[str, str]:
            """Normalize medical terminology in results."""
            try:
                json_data = json.loads(json_result)

                for page in json_data:
                    for region in page:
                        content = region.get("content", "")
                        if content:
                            # Normalize temperature
                            content = re.sub(
                                self.MEDICAL_PATTERNS["temperature"],
                                lambda m: f"{m.group(1)}°{m.group(2).replace('°', '')}",
                                content
                            )

                            # Normalize blood pressure
                            content = re.sub(
                                self.MEDICAL_PATTERNS["blood_pressure"],
                                lambda m: f"{m.group(1)}/{m.group(2)} mmHg",
                                content
                            )

                            region["content"] = content

                json_result = json.dumps(json_data, ensure_ascii=False)

                # Also apply to markdown
                for pattern_name, pattern in self.MEDICAL_PATTERNS.items():
                    if pattern_name == "temperature":
                        markdown_result = re.sub(
                            pattern,
                            lambda m: f"{m.group(1)}°{m.group(2).replace('°', '')}",
                            markdown_result
                        )
                    elif pattern_name == "blood_pressure":
                        markdown_result = re.sub(
                            pattern,
                            lambda m: f"{m.group(1)}/{m.group(2)} mmHg",
                            markdown_result
                        )

            except Exception as e:
                logger.warning(f"Medical terminology normalization failed: {e}")

            return json_result, markdown_result

        def _clean_medical_text(
            self,
            json_result: str,
            markdown_result: str,
            context: Dict[str, Any] = None
        ) -> Tuple[str, str]:
            """Clean medical text of common OCR artifacts."""
            try:
                json_data = json.loads(json_result)

                for page in json_data:
                    for region in page:
                        content = region.get("content", "")
                        if content:
                            # Remove extra whitespace
                            content = re.sub(r'\s+', ' ', content).strip()

                            # Fix common OCR errors in medical context
                            content = content.replace('l', '1')  # lowercase L to 1
                            content = content.replace('O', '0')  # uppercase O to 0
                            content = content.replace('o', '0')  # lowercase o to 0

                            region["content"] = content

                json_result = json.dumps(json_data, ensure_ascii=False)

                # Also apply to markdown
                markdown_result = re.sub(r'\s+', ' ', markdown_result).strip()
                markdown_result = markdown_result.replace('l', '1')
                markdown_result = markdown_result.replace('O', '0')
                markdown_result = markdown_result.replace('o', '0')

            except Exception as e:
                logger.warning(f"Medical text cleaning failed: {e}")

            return json_result, markdown_result

        def _apply_preprocessing(self, image, context: Dict[str, Any] = None):
            """Apply all registered preprocessing hooks to an image."""
            processed = image
            for hook in self.preprocess_hooks:
                try:
                    processed = hook(processed, context)
                except Exception as e:
                    logger.warning(f"Preprocessing hook failed: {e}")
            return processed

        def _apply_postprocessing(
            self,
            json_result: str,
            markdown_result: str,
            context: Dict[str, Any] = None
        ) -> Tuple[str, str]:
            """Apply all registered postprocessing hooks to results."""
            json_res, md_res = json_result, markdown_result
            for hook in self.postprocess_hooks:
                try:
                    json_res, md_res = hook(json_res, md_res, context)
                except Exception as e:
                    logger.warning(f"Postprocessing hook failed: {e}")
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
            """Process request with medical OCR enhancements."""
            preprocess_context = preprocess_context or {}
            postprocess_context = postprocess_context or {}

            logger.info("Processing with MedicalOcrPipeline")

            # Process with parent pipeline
            for result in super().process(
                request_data,
                save_layout_visualization=save_layout_visualization,
                page_maxsize=page_maxsize,
                region_maxsize=region_maxsize,
                preserve_order=preserve_order,
            ):
                # Apply postprocessing
                json_result, markdown_result = self._apply_postprocessing(
                    result.json_result,
                    result.markdown_result,
                    postprocess_context
                )

                # Create enhanced result
                enhanced_result = PipelineResult(
                    json_result=json_result,
                    markdown_result=markdown_result,
                    original_images=result.original_images,
                    image_files=result.image_files,
                    raw_json_result=result.raw_json_result,
                    layout_vis_images=result.layout_vis_images,
                )

                yield enhanced_result

