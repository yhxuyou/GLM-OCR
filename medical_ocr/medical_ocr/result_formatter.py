"""Medical Result Formatter - Specialized post-processing for medical documents.

Inherits from glmocr's ResultFormatter and adds RapidOCR-based coordinate matching.
"""

import json
import re
from typing import List, Dict, Tuple, Any, Optional

from glmocr.postprocess import ResultFormatter
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class MedicalResultFormatter(ResultFormatter):
    """Medical-specific result formatter with RapidOCR coordinate matching.

    Features:
    - Initialize RapidOCR model for text detection
    - Match OCR results with detected text coordinates
    - Output structured results with coordinates
    """

    def __init__(self, config):
        super().__init__(config)
        
        # RapidOCR model instances
        self._rapidocr_detector = None
        self._rapidocr_recognizer = None
        self._rapidocr_reader = None
        
        # Initialize RapidOCR models
        self._init_rapidocr()

    def _init_rapidocr(self):
        """Initialize RapidOCR models for text detection and recognition."""
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._rapidocr_reader = RapidOCR()
            logger.info("RapidOCR initialized successfully for coordinate detection")
        except ImportError:
            logger.warning("rapidocr_onnxruntime not installed, coordinate matching disabled")
        except Exception as e:
            logger.error(f"Failed to initialize RapidOCR: {e}")

    def _detect_text_with_coordinates(
        self, 
        image: Any
    ) -> List[Dict[str, Any]]:
        """Detect text regions and their coordinates using RapidOCR.

        Args:
            image: PIL Image or numpy array

        Returns:
            List of detected text regions with coordinates
        """
        if self._rapidocr_reader is None:
            return []

        try:
            # Convert PIL Image to numpy if needed
            from PIL import Image
            import numpy as np
            
            if isinstance(image, Image.Image):
                img_np = np.array(image.convert("RGB"))
            else:
                img_np = image

            # Run RapidOCR detection and recognition
            result, _, _ = self._rapidocr_reader(img_np)
            
            if result is None:
                return []

            # Parse results
            detected_texts = []
            for item in result:
                # item format: [box, text, score]
                # box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                box = item[0]
                text = item[1]
                score = item[2]
                
                # Calculate bounding box
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                
                # Normalize coordinates (0-1000 scale)
                height, width = img_np.shape[:2]
                x1_norm = int(x1 / width * 1000)
                y1_norm = int(y1 / height * 1000)
                x2_norm = int(x2 / width * 1000)
                y2_norm = int(y2 / height * 1000)
                
                detected_texts.append({
                    "text": text,
                    "score": float(score),
                    "bbox_2d": [x1_norm, y1_norm, x2_norm, y2_norm],
                    "polygon": [[int(p[0]), int(p[1])] for p in box],
                })
            
            return detected_texts

        except Exception as e:
            logger.warning(f"Text detection failed: {e}")
            return []

    def _match_ocr_with_coordinates(
        self, 
        ocr_content: str, 
        detected_texts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Match OCR content with detected text coordinates.

        Args:
            ocr_content: Raw OCR text content
            detected_texts: Detected text regions with coordinates

        Returns:
            List of matched results with text and coordinates
        """
        if not detected_texts:
            return [{"text": ocr_content, "bbox_2d": None, "polygon": None}]

        matched_results = []
        
        # Sort detected texts by position (top to bottom, left to right)
        sorted_texts = sorted(
            detected_texts, 
            key=lambda x: (x["bbox_2d"][1], x["bbox_2d"][0])
        )
        
        # Build matched results
        for det in sorted_texts:
            matched_results.append({
                "text": det["text"],
                "score": det["score"],
                "bbox_2d": det["bbox_2d"],
                "polygon": det["polygon"],
            })
        
        return matched_results

    def format_ocr_result(
        self, 
        content: str, 
        page_idx: int = 0,
        image: Any = None
    ) -> Tuple[str, str]:
        """Format OCR result with coordinate information.

        Args:
            content: Raw OCR output text
            page_idx: Page index
            image: PIL Image for coordinate detection (optional)

        Returns:
            (json_str, markdown_str) where json_str includes coordinate information
        """
        # Clean content
        content = self._clean_content(content)

        # Detect text coordinates if image is provided
        detected_texts = []
        if image is not None:
            detected_texts = self._detect_text_with_coordinates(image)

        # Match OCR content with coordinates
        if detected_texts:
            # Use coordinate-matched results
            matched_results = self._match_ocr_with_coordinates(content, detected_texts)
            
            # Build JSON with coordinates
            json_result = [
                {
                    "index": i,
                    "label": "text",
                    "content": item["text"],
                    "score": item.get("score", 1.0),
                    "bbox_2d": item["bbox_2d"],
                    "polygon": item.get("polygon"),
                }
                for i, item in enumerate(matched_results)
            ]
            
            # Build markdown
            markdown_result = "\n".join(item["text"] for item in matched_results)
        else:
            # Fallback to standard format without coordinates
            json_result = [
                {
                    "index": 0,
                    "label": "text",
                    "content": content,
                    "bbox_2d": None,
                }
            ]
            markdown_result = content

        json_str = json.dumps(json_result, ensure_ascii=False)
        return json_str, markdown_result

    def process(
        self,
        grouped_results: List[List[Dict]],
        cropped_images: Dict[tuple, Any] | None = None,
        image_prefix: str = "cropped",
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Process grouped results with coordinate enhancement.

        Args:
            grouped_results: Region recognition results grouped by page
            cropped_images: Pre-cropped images with coordinates
            image_prefix: Filename prefix for saved images

        Returns:
            (json_str, markdown_str, image_files)
        """
        # Call parent process first
        json_str, markdown_str, image_files = super().process(
            grouped_results,
            cropped_images=cropped_images,
            image_prefix=image_prefix,
        )
        
        # Parse and enhance with coordinates if images are available
        if cropped_images:
            json_data = json.loads(json_str)
            enhanced_json = []
            
            for page_idx, page_results in enumerate(json_data):
                enhanced_page = []
                for region in page_results:
                    region_copy = region.copy()
                    
                    # Try to find matching image
                    bbox = region.get("bbox_2d", [])
                    if bbox and cropped_images:
                        key = (page_idx, *bbox) if bbox else None
                        img = cropped_images.get(key) if key else None
                        
                        if img is not None:
                            # Detect text with coordinates in this region
                            detected = self._detect_text_with_coordinates(img)
                            if detected:
                                # Combine detected texts
                                combined_text = " ".join(d["text"] for d in detected)
                                combined_bbox = self._merge_bboxes([d["bbox_2d"] for d in detected])
                                combined_polygon = self._merge_polygons([d.get("polygon", []) for d in detected])
                                
                                region_copy["content"] = combined_text
                                region_copy["bbox_2d"] = combined_bbox
                                region_copy["polygon"] = combined_polygon
                                region_copy["detected_count"] = len(detected)
                    
                    enhanced_page.append(region_copy)
                enhanced_json.append(enhanced_page)
            
            json_str = json.dumps(enhanced_json, ensure_ascii=False)
        
        return json_str, markdown_str, image_files

    def _merge_bboxes(self, bboxes: List[List[int]]) -> List[int]:
        """Merge multiple bounding boxes into one.

        Args:
            bboxes: List of [x1, y1, x2, y2] boxes

        Returns:
            Merged [x1, y1, x2, y2] box
        """
        if not bboxes:
            return [0, 0, 1000, 1000]
        
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        
        return [x1, y1, x2, y2]

    def _merge_polygons(self, polygons: List[List[List[int]]]) -> List[List[int]]:
        """Merge multiple polygons into one.

        Args:
            polygons: List of polygon point lists

        Returns:
            Merged polygon points
        """
        if not polygons:
            return [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
        
        # Flatten and find bounds
        all_points = [p for poly in polygons for p in poly]
        if not all_points:
            return [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
        
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        return [
            [min(x_coords), min(y_coords)],
            [max(x_coords), min(y_coords)],
            [max(x_coords), max(y_coords)],
            [min(x_coords), max(y_coords)],
        ]
