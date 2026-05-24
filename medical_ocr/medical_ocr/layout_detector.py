"""Medical Layout Detector - Specialized for medical documents.

Inherits from glmocr's PPDocLayoutDetector.
Keeps only 'table' regions from parent detection and adds the full image as 'text'.
"""

from typing import List, Dict, Any, Tuple
from PIL import Image

from glmocr.layout import PPDocLayoutDetector
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class MedicalLayoutDetector(PPDocLayoutDetector):
    """Medical-specific layout detector.

    Process flow:
    1. Call parent's process() to get all detections
    2. Filter to keep only 'table' label regions
    3. Add the full image as a single 'text' region
    """

    def __init__(self, config):
        super().__init__(config)
        self.target_labels = ["table"]  # Labels to keep

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        global_start_idx: int = 0,
        use_polygon: bool = False,
    ) -> Tuple[List[List[Dict[str, Any]]], Dict[int, Image.Image]]:
        """Process images with medical-specific filtering.

        Args:
            images: List of PIL Images.
            save_visualization: Whether to generate visualization images.
            global_start_idx: Start index for visualization.
            use_polygon: Use polygon masks.

        Returns:
            Tuple of (results, vis_images)
        """
        logger.info(f"Processing {len(images)} images with medical layout detector")

        # Step 1: Call parent class process
        all_results, vis_images = super().process(
            images,
            save_visualization=save_visualization,
            global_start_idx=global_start_idx,
            use_polygon=use_polygon,
        )

        # Step 2: Filter and add full image region
        final_results = []
        for img_idx, page_results in enumerate(all_results):
            image_width, image_height = images[img_idx].size

            # Filter: keep only 'table' label regions
            filtered = []
            for region in page_results:
                label = region.get("label", "").lower()
                if label in self.target_labels:
                    filtered.append(region)

            # Add full image as 'text' region
            full_image_region = {
                "index": len(filtered),
                "label": "text",
                "score": 1.0,
                "bbox_2d": [0, 0, 1000, 1000],
                "polygon": [
                    [0, 0],
                    [1000, 0],
                    [1000, 1000],
                    [0, 1000],
                ],
                "task_type": "text",
            }
            filtered.append(full_image_region)

            final_results.append(filtered)

            logger.debug(
                f"Page {img_idx + global_start_idx}: "
                f"{len(page_results)} -> {len(filtered)} regions "
                f"(kept tables + full image as text)"
            )

        return final_results, vis_images
