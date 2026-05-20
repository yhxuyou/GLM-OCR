"""Custom Layout Detector Example - Demonstrates inheritance.

This file shows how to create your own subclass that inherits from MedicalLayoutDetector
and implements custom post-processing logic.
"""

from typing import List, Dict, Any, Tuple
from PIL import Image

from medical_ocr.layout_detector import MedicalLayoutDetector
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class CustomMedicalLayoutDetector(MedicalLayoutDetector):
    """Example custom layout detector.

    Demonstrates how to:
    1. Inherit from MedicalLayoutDetector
    2. Add custom initialization parameters
    3. Override process() to add custom post-processing
    4. Add new methods for custom logic
    """

    def __init__(self, config, custom_param: str = "default"):
        """Initialize custom layout detector.

        Args:
            config: LayoutConfig instance.
            custom_param: Custom parameter for your logic.
        """
        super().__init__(config)
        
        # Add your custom initialization here
        self.custom_param = custom_param
        self.keep_labels = ["table", "title"]  # Custom labels to keep
        self.min_region_area = 100  # Minimum region area (normalized)
        
        logger.info(f"CustomMedicalLayoutDetector initialized with param: {custom_param}")

    def _filter_small_regions(self, regions: List[Dict]) -> List[Dict]:
        """Custom filter: remove small regions.

        Args:
            regions: List of region dicts.

        Returns:
            Filtered list with small regions removed.
        """
        filtered = []
        for region in regions:
            bbox = region.get("bbox_2d", [0, 0, 0, 0])
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            if area >= self.min_region_area:
                filtered.append(region)
        
        logger.debug(f"Filtered {len(regions)} -> {len(filtered)} regions (area threshold)")
        return filtered

    def _add_custom_regions(self, page_results: List[Dict], image_width: int, image_height: int) -> List[Dict]:
        """Add custom regions to the results.

        Args:
            page_results: Current region list.
            image_width: Original image width.
            image_height: Original image height.

        Returns:
            List with custom regions added.
        """
        # Example: Add a header region (top 10% of image)
        header_region = {
            "index": len(page_results),
            "label": "header",
            "score": 0.95,
            "bbox_2d": [0, 0, 1000, 100],  # Top 10% height (normalized)
            "polygon": [
                [0, 0],
                [1000, 0],
                [1000, 100],
                [0, 100],
            ],
            "task_type": "text",
            "custom_tag": "header_region",
        }
        page_results.append(header_region)

        # Example: Add a footer region (bottom 10% of image)
        footer_region = {
            "index": len(page_results),
            "label": "footer",
            "score": 0.95,
            "bbox_2d": [0, 900, 1000, 1000],  # Bottom 10% height
            "polygon": [
                [0, 900],
                [1000, 900],
                [1000, 1000],
                [0, 1000],
            ],
            "task_type": "text",
            "custom_tag": "footer_region",
        }
        page_results.append(footer_region)

        return page_results

    def _custom_postprocess(self, page_results: List[Dict]) -> List[Dict]:
        """Apply your custom post-processing logic.

        This is where you can add any custom logic.

        Args:
            page_results: List of region dicts.

        Returns:
            Post-processed regions.
        """
        # Example custom processing: sort by y-coordinate
        page_results.sort(key=lambda r: r["bbox_2d"][1])
        
        # Example: Add custom metadata
        for i, region in enumerate(page_results):
            region["custom_index"] = i
            region["is_custom_processed"] = True
        
        return page_results

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        global_start_idx: int = 0,
        use_polygon: bool = False,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[int, Image.Image]]:
        """Process images with custom post-processing.

        Override this method to add your custom logic.

        Args:
            images: List of PIL Images.
            save_visualization: Whether to generate visualization images.
            global_start_idx: Start index for visualization.
            use_polygon: Use polygon masks.

        Returns:
            Tuple of (results, vis_images)
        """
        logger.info(f"Processing {len(images)} images with CustomMedicalLayoutDetector")

        # Step 1: Call parent's process (which calls grandparent's process)
        # This gets us the filtered results (tables + full image)
        all_results, vis_images = super().process(
            images,
            save_visualization=save_visualization,
            global_start_idx=global_start_idx,
            use_polygon=use_polygon,
        )

        # Step 2: Apply your custom post-processing
        final_results = []
        for img_idx, page_results in enumerate(all_results):
            image_width, image_height = images[img_idx].size

            # Custom step 1: Filter small regions
            page_results = self._filter_small_regions(page_results)

            # Custom step 2: Add custom regions (header, footer)
            page_results = self._add_custom_regions(page_results, image_width, image_height)

            # Custom step 3: Apply any additional post-processing
            page_results = self._custom_postprocess(page_results)

            # Update indices to maintain order
            for i, region in enumerate(page_results):
                region["index"] = i

            final_results.append(page_results)

            logger.debug(
                f"Page {img_idx + global_start_idx}: "
                f"Custom processing completed, {len(page_results)} regions"
            )

        return final_results, vis_images


# ====================
# Another Example: Specialized Report Detector
# ====================

class MedicalReportLayoutDetector(MedicalLayoutDetector):
    """Specialized detector for medical reports.

    Focuses on detecting specific medical report elements.
    """

    MEDICAL_REPORT_LABELS = ["table", "title", "section_header", "signature"]

    def __init__(self, config):
        super().__init__(config)
        self.target_labels = self.MEDICAL_REPORT_LABELS

    def _detect_signature_region(self, page_results: List[Dict], image_height: int) -> List[Dict]:
        """Detect potential signature region (bottom-right corner)."""
        signature_region = {
            "index": len(page_results),
            "label": "signature",
            "score": 0.8,
            "bbox_2d": [700, 800, 1000, 1000],  # Bottom-right corner
            "polygon": [
                [700, 800],
                [1000, 800],
                [1000, 1000],
                [700, 1000],
            ],
            "task_type": "text",
            "is_signature_candidate": True,
        }
        return signature_region

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        global_start_idx: int = 0,
        use_polygon: bool = False,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[int, Image.Image]]:
        """Process medical report images."""
        logger.info(f"Processing {len(images)} medical report images")

        # Call parent process
        all_results, vis_images = super().process(
            images,
            save_visualization=save_visualization,
            global_start_idx=global_start_idx,
            use_polygon=use_polygon,
        )

        # Add signature detection
        final_results = []
        for img_idx, page_results in enumerate(all_results):
            image_height = images[img_idx].size[1]
            
            # Add signature region candidate
            sig_region = self._detect_signature_region(page_results, image_height)
            page_results.append(sig_region)
            
            # Update indices
            for i, region in enumerate(page_results):
                region["index"] = i
            
            final_results.append(page_results)

        return final_results, vis_images


# ====================
# Usage Example
# ====================

def example_usage():
    """Example showing how to use custom layout detectors."""
    from glmocr.config import load_config

    config = load_config()

    # Create custom detector
    custom_detector = CustomMedicalLayoutDetector(
        config.pipeline.layout,
        custom_param="my_custom_value"
    )

    # Create specialized detector
    report_detector = MedicalReportLayoutDetector(config.pipeline.layout)

    # Start detectors
    custom_detector.start()
    report_detector.start()

    # Use them
    # ... (process images)

    # Stop
    custom_detector.stop()
    report_detector.stop()


if __name__ == "__main__":
    example_usage()
