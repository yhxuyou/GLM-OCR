"""Optimized PP-DocLayoutV3 layout detector with enhanced pre/post processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict

import cv2
import numpy as np
from PIL import Image

from glmocr.layout.layout_detector import PPDocLayoutDetector
from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.config import LayoutConfig

logger = get_logger(__name__)


class OptimizedPPDocLayoutDetector(PPDocLayoutDetector):
    """Optimized PP-DocLayoutV3 layout detector with enhanced processing.
    
    Features:
    - Enhanced preprocessing: noise reduction, contrast enhancement
    - Improved region filtering: remove overlapping regions, merge small regions
    - Better polygon handling: smooth polygons, fix invalid shapes
    """

    def __init__(self, config: "LayoutConfig"):
        super().__init__(config)
        self.enable_preprocessing = getattr(config, "enable_preprocessing", True)
        self.enable_region_filtering = getattr(config, "enable_region_filtering", True)
        self.min_region_area = getattr(config, "min_region_area", 200)
        self.max_overlap_ratio = getattr(config, "max_overlap_ratio", 0.8)

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality before layout detection."""
        if not self.enable_preprocessing:
            return image

        img_np = np.array(image.convert("RGB"))
        
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
        
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        return Image.fromarray(img_np)

    def _filter_overlapping_regions(self, regions: List[Dict]) -> List[Dict]:
        """Remove highly overlapping regions, keeping the one with higher confidence."""
        if not self.enable_region_filtering or len(regions) < 2:
            return regions

        def calculate_iou(bbox1, bbox2):
            x1, y1, x2, y2 = bbox1
            x1b, y1b, x2b, y2b = bbox2
            
            inter_x1 = max(x1, x1b)
            inter_y1 = max(y1, y1b)
            inter_x2 = min(x2, x2b)
            inter_y2 = min(y2, y2b)
            
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x2b - x1b) * (y2b - y1b)
            
            return inter_area / min(area1, area2)

        regions = sorted(regions, key=lambda r: r.get("score", 0), reverse=True)
        filtered = []
        
        for region in regions:
            bbox = region["bbox_2d"]
            keep = True
            
            for existing in filtered:
                existing_bbox = existing["bbox_2d"]
                iou = calculate_iou(bbox, existing_bbox)
                if iou > self.max_overlap_ratio:
                    keep = False
                    break
            
            if keep:
                filtered.append(region)
        
        return filtered

    def _filter_small_regions(self, regions: List[Dict], image_width: int, image_height: int) -> List[Dict]:
        """Remove regions that are too small."""
        if not self.enable_region_filtering:
            return regions

        filtered = []
        for region in regions:
            bbox = region["bbox_2d"]
            x1, y1, x2, y2 = bbox
            
            width_px = (x2 - x1) * image_width / 1000
            height_px = (y2 - y1) * image_height / 1000
            area = width_px * height_px
            
            if area >= self.min_region_area:
                filtered.append(region)
        
        return filtered

    def _smooth_polygon(self, polygon: List[List[int]]) -> List[List[int]]:
        """Apply light smoothing to polygon points."""
        if len(polygon) < 4:
            return polygon
        
        polygon_np = np.array(polygon, dtype=np.float32)
        kernel = np.ones((3,)) / 3.0
        
        x_coords = np.convolve(polygon_np[:, 0], kernel, mode='same')
        y_coords = np.convolve(polygon_np[:, 1], kernel, mode='same')
        
        smoothed = [[int(x_coords[i]), int(y_coords[i])] for i in range(len(polygon))]
        
        return smoothed

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        global_start_idx: int = 0,
        use_polygon: bool = False,
    ) -> tuple:
        """Batch-detect layout regions with enhanced pre/post processing."""
        if self._model is None:
            raise RuntimeError("Layout detector not started. Call start() first.")
        self._validate_runtime_config()

        num_images = len(images)
        
        processed_images = [
            self._preprocess_image(img) for img in images
        ]
        
        results, vis_images = super().process(
            processed_images,
            save_visualization=save_visualization,
            global_start_idx=global_start_idx,
            use_polygon=use_polygon,
        )
        
        enhanced_results = []
        for img_idx, page_results in enumerate(results):
            original_image = images[img_idx]
            image_width, image_height = original_image.size
            
            page_results = self._filter_small_regions(page_results, image_width, image_height)
            page_results = self._filter_overlapping_regions(page_results)
            
            for region in page_results:
                if "polygon" in region and region["polygon"]:
                    region["polygon"] = self._smooth_polygon(region["polygon"])
            
            enhanced_results.append(page_results)
        
        return enhanced_results, vis_images