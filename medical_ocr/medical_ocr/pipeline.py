"""Medical OCR Pipeline - Simplified processing pipeline.

This module provides a simplified pipeline that uses MedicalPageLoader
for preprocessing and integrates with glmocr's OCR capabilities.
"""

from typing import Dict, Any, List, Generator, Optional, Callable, Tuple
import json
import re

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class MedicalOcrPipeline:
    """Simplified Medical OCR Pipeline.

    This pipeline focuses on preprocessing using MedicalPageLoader
    and provides hooks for custom processing.

    Features:
    - Medical document preprocessing (YOLO + RapidOCR + UVDoc)
    - Preprocessing and postprocessing hooks
    - Medical terminology post-processing
    """

    # Medical terminology patterns for post-processing
    MEDICAL_PATTERNS = {
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
        page_loader,
        ocr_client=None,
        layout_detector=None,
        result_formatter=None,
    ):
        """Initialize MedicalOcrPipeline.

        Args:
            page_loader: MedicalPageLoader instance for preprocessing
            ocr_client: OCR client for recognition (optional)
            layout_detector: Layout detector (optional)
            result_formatter: Result formatter (optional)
        """
        self.page_loader = page_loader
        self.ocr_client = ocr_client
        self.layout_detector = layout_detector
        self.result_formatter = result_formatter

        # Hook lists
        self.preprocess_hooks: List[Callable] = []
        self.postprocess_hooks: List[Callable] = []

        # Register default hooks
        self._register_default_hooks()

    def _register_default_hooks(self):
        """Register default postprocessing hooks."""
        self.add_postprocess_hook(self._medical_terminology_normalization)
        self.add_postprocess_hook(self._clean_medical_text)

    def add_preprocess_hook(self, hook: Callable):
        """Add a custom preprocessing hook.

        Args:
            hook: Function that takes (image, context) and returns processed image
        """
        self.preprocess_hooks.append(hook)

    def add_postprocess_hook(self, hook: Callable):
        """Add a custom postprocessing hook.

        Args:
            hook: Function that takes (json_result, markdown_result, context)
                  and returns (json_result, markdown_result)
        """
        self.postprocess_hooks.append(hook)

    def _medical_terminology_normalization(
        self,
        json_result: str,
        markdown_result: str,
        context: Dict[str, Any] = None
    ) -> Tuple[str, str]:
        """Normalize medical terminology in results."""
        try:
            json_data = json.loads(json_result) if json_result else []

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
            json_data = json.loads(json_result) if json_result else []

            for page in json_data:
                for region in page:
                    content = region.get("content", "")
                    if content:
                        # Remove extra whitespace
                        content = re.sub(r'\s+', ' ', content).strip()

                        # Fix common OCR errors in medical context
                        content = content.replace('l', '1')
                        content = content.replace('O', '0')
                        content = content.replace('o', '0')

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

    def start(self):
        """Start the pipeline components."""
        logger.info("Starting MedicalOcrPipeline...")

        if self.page_loader is not None:
            if hasattr(self.page_loader, 'start'):
                self.page_loader.start()
                logger.info("MedicalPageLoader started")

        if self.layout_detector is not None:
            if hasattr(self.layout_detector, 'start'):
                self.layout_detector.start()
                logger.info("Layout detector started")

        if self.ocr_client is not None:
            if hasattr(self.ocr_client, 'start'):
                self.ocr_client.start()
                logger.info("OCR client started")

        logger.info("MedicalOcrPipeline started")

    def stop(self):
        """Stop the pipeline components."""
        logger.info("Stopping MedicalOcrPipeline...")

        if self.ocr_client is not None:
            if hasattr(self.ocr_client, 'stop'):
                self.ocr_client.stop()

        if self.layout_detector is not None:
            if hasattr(self.layout_detector, 'stop'):
                self.layout_detector.stop()

        if self.page_loader is not None:
            if hasattr(self.page_loader, 'stop'):
                self.page_loader.stop()

        logger.info("MedicalOcrPipeline stopped")

    def process_images(
        self,
        images: List[Any],
        preprocess_context: Optional[Dict[str, Any]] = None,
        postprocess_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Process a list of images through the pipeline.

        Args:
            images: List of PIL Images
            preprocess_context: Context for preprocessing hooks
            postprocess_context: Context for postprocessing hooks

        Returns:
            List of processed results
        """
        preprocess_context = preprocess_context or {}
        postprocess_context = postprocess_context or {}

        results = []

        for image in images:
            # Apply preprocessing hooks
            processed = self._apply_preprocessing(image, preprocess_context)

            # Process image (layout detection + OCR)
            result = self._process_single_image(processed)
            
            # Apply postprocessing hooks
            json_result, markdown_result = self._apply_postprocessing(
                result.get("json_result", ""),
                result.get("markdown_result", ""),
                postprocess_context
            )
            
            result["json_result"] = json_result
            result["markdown_result"] = markdown_result
            results.append(result)

        return results

    def _process_single_image(self, image) -> Dict[str, Any]:
        """Process a single image.

        Args:
            image: PIL Image

        Returns:
            Dict with json_result and markdown_result
        """
        # Placeholder for actual processing
        # In production, this would call layout_detector and ocr_client
        
        return {
            "json_result": "[]",
            "markdown_result": "",
            "image": image,
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
