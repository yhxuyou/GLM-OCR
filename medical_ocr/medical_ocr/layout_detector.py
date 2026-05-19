"""Medical Layout Detector - Specialized for medical documents.

Inherits from glmocr's PPDocLayoutDetector and adds medical-specific optimizations.
"""

from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)

# 可选依赖
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("cv2/numpy not available, some enhancement features will be disabled")

# 导入基础布局检测器
try:
    from glmocr.layout import PPDocLayoutDetector
    BASE_DETECTOR_AVAILABLE = True
except Exception as e:
    logger.warning(f"PPDocLayoutDetector not available: {e}")
    PPDocLayoutDetector = None
    BASE_DETECTOR_AVAILABLE = False


# 只有在基础检测器可用时才定义我们的类
if BASE_DETECTOR_AVAILABLE and PPDocLayoutDetector:
    class MedicalLayoutDetector(PPDocLayoutDetector):
        """Specialized layout detector for medical documents.

        Features:
        - Medical document specific preprocessing
        - Enhanced detection of medical elements (tables, charts, signatures)
        - Post-filtering for medical-specific regions
        - Better handling of low-quality medical scans
        """

        # Medical document specific region labels to prioritize
        MEDICAL_LABELS = {
            "table", "chart", "diagram", "signature",
            "stamp", "header", "footer", "medical_record"
        }

        def __init__(self, config):
            super().__init__(config)

            # Medical-specific configuration
            self.enable_medical_enhancement = getattr(
                config, "enable_medical_enhancement", True
            )
            self.medical_threshold_multiplier = getattr(
                config, "medical_threshold_multiplier", 0.85
            )
            self.min_medical_region_area = getattr(
                config, "min_medical_region_area", 300
            )
            self.max_overlap_ratio = getattr(
                config, "max_overlap_ratio", 0.85
            )

        def _enhance_medical_image(self, image: Image.Image) -> Image.Image:
            """Apply medical-specific image enhancement.

            Optimizes for low-quality medical scans:
            - Noise reduction
            - Contrast enhancement
            - Edge sharpening
            """
            if not CV2_AVAILABLE or not self.enable_medical_enhancement:
                return image

            try:
                img_np = np.array(image.convert("RGB"))

                # Convert to grayscale for some processing
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                # Denoising - Non-local means for better medical image quality
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

                # CLAHE for contrast enhancement (good for x-rays and scans)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                contrast_enhanced = clahe.apply(denoised)

                # Convert back to RGB
                enhanced_rgb = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2RGB)

                # Combine with original for better color
                alpha = 0.7
                final = cv2.addWeighted(enhanced_rgb, alpha, img_np, 1 - alpha, 0)

                return Image.fromarray(final)
            except Exception as e:
                logger.warning(f"Medical enhancement failed: {e}, using original image")
                return image

        def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
            """Calculate Intersection over Union for two boxes."""
            x1, y1, x2, y2 = box1
            x3, y3, x4, y4 = box2

            inter_x1 = max(x1, x3)
            inter_y1 = max(y1, y3)
            inter_x2 = min(x2, x4)
            inter_y2 = min(y2, y4)

            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x4 - x3) * (y4 - y3)

            return inter_area / min(area1, area2)

        def _filter_medical_regions(
            self,
            regions: List[Dict[str, Any]],
            img_width: int,
            img_height: int
        ) -> List[Dict[str, Any]]:
            """Filter and prioritize medical-specific regions."""
            if not regions:
                return []

            filtered = []

            for region in regions:
                bbox = region.get("bbox_2d", [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox

                # Calculate actual pixel area
                width = (x2 - x1) * img_width / 1000
                height = (y2 - y1) * img_height / 1000
                area = width * height

                # Skip too small regions
                if area < self.min_medical_region_area:
                    continue

                # Adjust confidence for medical labels
                label = region.get("label", "").lower()
                score = region.get("score", 0.0)

                if any(med_label in label for med_label in self.MEDICAL_LABELS):
                    # Boost confidence for medical-specific labels
                    region["score"] = min(1.0, score * 1.2)
                    region["is_medical_region"] = True

                filtered.append(region)

            # Remove overlapping regions, keep higher confidence
            final_regions = []
            filtered.sort(key=lambda r: r.get("score", 0), reverse=True)

            for region in filtered:
                bbox = region.get("bbox_2d", [0, 0, 0, 0])
                keep = True

                for existing in final_regions:
                    existing_bbox = existing.get("bbox_2d", [0, 0, 0, 0])
                    iou = self._calculate_iou(bbox, existing_bbox)
                    if iou > self.max_overlap_ratio:
                        keep = False
                        break

                if keep:
                    final_regions.append(region)

            return final_regions

        def _smooth_polygons(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Smooth polygon coordinates for cleaner regions."""
            if not CV2_AVAILABLE:
                return regions

            for region in regions:
                polygon = region.get("polygon", [])
                if len(polygon) >= 4:
                    try:
                        # Apply moving average smoothing
                        smoothed = []
                        polygon_np = np.array(polygon, dtype=np.float32)
                        window_size = 3

                        for i in range(len(polygon_np)):
                            start = max(0, i - window_size // 2)
                            end = min(len(polygon_np), i + window_size // 2 + 1)
                            avg_point = np.mean(polygon_np[start:end], axis=0)
                            smoothed.append([int(avg_point[0]), int(avg_point[1])])

                        region["polygon"] = smoothed
                    except Exception as e:
                        logger.warning(f"Polygon smoothing failed: {e}")

            return regions

        def process(
            self,
            images: List[Image.Image],
            save_visualization: bool = False,
            global_start_idx: int = 0,
            use_polygon: bool = False,
        ) -> Tuple[List[List[Dict[str, Any]]], Dict[int, Image.Image]]:
            """Process images with medical-specific enhancements."""
            logger.info(f"Processing {len(images)} images with medical layout detector")

            # Step 1: Apply medical image enhancement
            enhanced_images = []
            if self.enable_medical_enhancement:
                for img in images:
                    try:
                        enhanced_img = self._enhance_medical_image(img)
                        enhanced_images.append(enhanced_img)
                    except Exception as e:
                        logger.warning(f"Image enhancement failed: {e}, using original")
                        enhanced_images.append(img)
            else:
                enhanced_images = images

            # Step 2: Call parent class process with enhanced images
            results, vis_images = super().process(
                enhanced_images,
                save_visualization=save_visualization,
                global_start_idx=global_start_idx,
                use_polygon=use_polygon,
            )

            # Step 3: Apply medical-specific post-processing
            final_results = []
            for img_idx, page_results in enumerate(results):
                original_img = images[img_idx]
                img_width, img_height = original_img.size

                # Filter and prioritize medical regions
                filtered_regions = self._filter_medical_regions(
                    page_results, img_width, img_height
                )

                # Smooth polygons if needed
                if use_polygon:
                    filtered_regions = self._smooth_polygons(filtered_regions)

                final_results.append(filtered_regions)

                logger.debug(
                    f"Page {img_idx + global_start_idx}: "
                    f"{len(page_results)} -> {len(filtered_regions)} regions after filtering"
                )

            return final_results, vis_images

