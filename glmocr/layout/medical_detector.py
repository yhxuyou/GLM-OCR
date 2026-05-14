"""
Medical Expense Layout Detector - 医疗费用清单布局检测器

专门针对医疗费用清单场景优化的布局检测器，支持：
- 表格区域识别和扩展裁剪（向外扩展5%，最小10像素）
- 非表格区域整体合并（保留相对空间位置）
- 结构化输出（表格裁剪图列表 + 非表格整体图）

作者: GLM-OCR
版本: 1.0.0
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessor,
)

from glmocr.layout.base import BaseLayoutDetector
from glmocr.utils.layout_postprocess_utils import apply_layout_postprocess
from glmocr.utils.logging import get_logger
from glmocr.utils.visualization_utils import draw_layout_boxes

if TYPE_CHECKING:
    from glmocr.config import LayoutConfig

logger = get_logger(__name__)


@dataclass
class TableRegion:
    """表格区域数据结构"""
    table_id: int
    bbox_2d: List[int]  # [x1, y1, x2, y2] 归一化坐标 (0-1000)
    bbox_pixel: Tuple[int, int, int, int]  # 像素坐标
    expanded_bbox_2d: Optional[List[int]] = None  # 扩展后归一化坐标
    expanded_bbox_pixel: Optional[Tuple[int, int, int, int]] = None  # 扩展后像素坐标
    label: str = "table"
    score: float = 1.0
    polygon: Optional[List[List[float]]] = None
    cropped_image: Optional[Image.Image] = None


@dataclass
class NonTableImage:
    """非表格区域整体图像"""
    merged_image: Image.Image
    original_size: Tuple[int, int]
    cropped_regions_count: int = 0


@dataclass
class MedicalExpenseResult:
    """医疗费用清单处理结果"""
    table_regions: List[TableRegion] = field(default_factory=list)
    non_table_image: Optional[NonTableImage] = None
    original_image: Optional[Image.Image] = None
    image_size: Tuple[int, int] = (0, 0)
    all_regions: Optional[List[Dict]] = None  # 所有原始布局检测结果

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "table_count": len(self.table_regions),
            "table_regions": [
                {
                    "table_id": t.table_id,
                    "bbox_2d": t.bbox_2d,
                    "bbox_pixel": list(t.bbox_pixel) if t.bbox_pixel else None,
                    "expanded_bbox_2d": t.expanded_bbox_2d,
                    "expanded_bbox_pixel": list(t.expanded_bbox_pixel) if t.expanded_bbox_pixel else None,
                    "label": t.label,
                    "score": t.score,
                    "has_cropped_image": t.cropped_image is not None
                }
                for t in self.table_regions
            ],
            "has_non_table_image": self.non_table_image is not None,
            "non_table_image_info": self.non_table_image.to_dict() if self.non_table_image else None,
            "image_size": self.image_size
        }

    def save(self, output_dir: str, format: str = "PNG") -> Dict[str, str]:
        """保存所有结果到指定目录

        Args:
            output_dir: 输出目录路径
            format: 图像格式，默认 PNG

        Returns:
            保存的文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # 保存表格区域
        for table in self.table_regions:
            if table.cropped_image:
                table_path = output_path / f"table_{table.table_id:03d}.{format.lower()}"
                table.cropped_image.save(table_path, format=format)
                saved_files[f"table_{table.table_id:03d}"] = str(table_path)

        # 保存非表格区域整体图
        if self.non_table_image and self.non_table_image.merged_image:
            non_table_path = output_path / f"non_table_regions.{format.lower()}"
            self.non_table_image.merged_image.save(non_table_path, format=format)
            saved_files["non_table_regions"] = str(non_table_path)

        # 保存原始图像
        if self.original_image:
            original_path = output_path / f"original.{format.lower()}"
            self.original_image.save(original_path, format=format)
            saved_files["original"] = str(original_path)

        return saved_files


class MedicalExpenseLayoutDetector(BaseLayoutDetector):
    """医疗费用清单布局检测器

    基于 PP-DocLayoutV3，增加医疗费用清单专属的表格扩展裁剪和非表格区域合并功能。
    """

    TABLE_EXPAND_RATIO = 0.05  # 默认扩展5%
    TABLE_MIN_EXPANSION = 10   # 默认最小扩展10像素
    TABLE_LABELS = ["table", "表格"]  # 表格标签列表
    FILL_COLOR = (255, 255, 255)  # 非表格区域中表格部分的填充色

    def __init__(self, config: "LayoutConfig"):
        """初始化医疗费用清单布局检测器

        Args:
            config: LayoutConfig 实例
        """
        super().__init__(config)

        self.model_dir = config.model_dir
        self.cuda_visible_devices = config.cuda_visible_devices
        self._config_device = config.device

        self.threshold = config.threshold
        self.threshold_by_class = config.threshold_by_class
        self.layout_nms = config.layout_nms
        self.layout_unclip_ratio = config.layout_unclip_ratio
        self.layout_merge_bboxes_mode = config.layout_merge_bboxes_mode
        self.batch_size = config.batch_size

        self.label_task_mapping = config.label_task_mapping
        self.id2label = getattr(config, "id2label", None)

        # 医疗费用清单专属配置
        self.table_expand_ratio = getattr(config, "table_expand_ratio", self.TABLE_EXPAND_RATIO)
        self.table_min_expansion = getattr(config, "table_min_expansion", self.TABLE_MIN_EXPANSION)
        self.table_labels = getattr(config, "table_labels", self.TABLE_LABELS)
        self.fill_color = getattr(config, "fill_color", self.FILL_COLOR)

        self._model = None
        self._image_processor = None
        self._device = None

    def _validate_runtime_config(self):
        """验证配置"""
        if not self.model_dir:
            raise ValueError(
                "pipeline.layout.model_dir is required for self-hosted layout "
                "detection. Set it to a local checkpoint directory or a Hugging "
                "Face model id such as 'PaddlePaddle/PP-DocLayoutV3_safetensors'."
            )

    def start(self):
        """加载模型和处理器"""
        logger.debug("Initializing Medical Expense Layout Detector (PP-DocLayoutV3)...")
        self._validate_runtime_config()

        self._image_processor = PPDocLayoutV3ImageProcessor.from_pretrained(self.model_dir)
        self._model = PPDocLayoutV3ForObjectDetection.from_pretrained(self.model_dir)
        self._model.eval()

        # 设备选择
        if self._config_device is not None:
            self._device = self._config_device
        elif torch.cuda.is_available() and self.cuda_visible_devices:
            self._device = f"cuda:{self.cuda_visible_devices}"
        else:
            self._device = "cpu"
        self._model = self._model.to(self._device)

        if self.id2label is None:
            self.id2label = getattr(self._model.config, "id2label", None)
        if self.id2label is None:
            raise RuntimeError("Missing id2label in both layout config and model config")

        # 修复上游的多边形提取函数
        self._patch_polygon_extractor()

        if self.label_task_mapping is None:
            logger.warning("layout.label_task_mapping is missing; defaulting all labels to text")
            self.label_task_mapping = {"text": list(self.id2label.values())}

        logger.debug(f"Medical Expense Layout Detector loaded on device: {self._device}")

    def stop(self):
        """卸载模型"""
        if self._model is not None:
            if self._device.startswith("cuda"):
                torch.cuda.empty_cache()
            self._model = None
        self._image_processor = None
        self._device = None
        logger.debug("Medical Expense Layout Detector stopped.")

    def _patch_polygon_extractor(self):
        """修复上游的多边形提取函数，防止空掩码错误"""
        def _safe_extract(boxes, masks, scale_ratio):
            scale_w, scale_h = scale_ratio[0] / 4, scale_ratio[1] / 4
            mask_h, mask_w = masks.shape[1:]
            polygon_points = []

            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i].astype(np.int32)
                box_w, box_h = x_max - x_min, y_max - y_min
                rect = np.array(
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    dtype=np.float32,
                )

                if box_w <= 0 or box_h <= 0:
                    polygon_points.append(rect)
                    continue

                x_start = int(round((x_min * scale_w).item()))
                x_end = int(round((x_max * scale_w).item()))
                x_start, x_end = np.clip([x_start, x_end], 0, mask_w)
                y_start = int(round((y_min * scale_h).item()))
                y_end = int(round((y_max * scale_h).item()))
                y_start, y_end = np.clip([y_start, y_end], 0, mask_h)

                cropped_mask = masks[i, y_start:y_end, x_start:x_end]
                if cropped_mask.size == 0:
                    polygon_points.append(rect)
                    continue

                resized = cv2.resize(
                    cropped_mask.astype(np.uint8),
                    (box_w, box_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                polygon = self._image_processor._mask2polygon(resized)
                if polygon is not None and len(polygon) < 4:
                    polygon_points.append(rect)
                    continue
                if polygon is not None and len(polygon) > 0:
                    polygon = polygon + np.array([x_min, y_min])
                polygon_points.append(polygon)

            return polygon_points

        self._image_processor._extract_polygon_points_by_masks = _safe_extract

    def _apply_per_class_threshold(self, raw_results: List[Dict]):
        """按类别阈值过滤检测结果"""
        label2id = {name: int(cls_id) for cls_id, name in self.id2label.items()}

        class_thresholds = {}
        for key, value in self.threshold_by_class.items():
            if isinstance(key, str):
                if key in label2id:
                    class_thresholds[label2id[key]] = float(value)
                else:
                    logger.warning(
                        "Unknown class name '%s' in threshold_by_class; ignored.",
                        key
                    )
            else:
                class_thresholds[int(key)] = float(value)

        fallback = self.threshold
        filtered = []
        for result in raw_results:
            scores = result["scores"]
            labels = result["labels"]

            thresholds = torch.full_like(scores, fallback)
            for class_id, thresh in class_thresholds.items():
                thresholds[labels == class_id] = thresh

            keep = scores >= thresholds

            new_result = {
                "scores": scores[keep],
                "labels": labels[keep],
                "boxes": result["boxes"][keep],
            }
            if "order_seq" in result:
                new_result["order_seq"] = result["order_seq"][keep]
            if "polygon_points" in result:
                keep_list = keep.tolist()
                new_result["polygon_points"] = [
                    p for p, k in zip(result["polygon_points"], keep_list) if k
                ]
            filtered.append(new_result)
        return filtered

    def _empty_detection_result(self) -> Dict:
        """返回空的检测结果"""
        return {
            "scores": torch.tensor([], device=self._device),
            "labels": torch.tensor([], dtype=torch.long, device=self._device),
            "boxes": torch.tensor([], device=self._device).reshape(0, 4),
            "order_seq": torch.tensor([], dtype=torch.long, device=self._device),
        }

    def _run_detection_single_image(self, image: Image.Image, pre_threshold: float) -> Dict:
        """对单张图像进行检测"""
        single_inputs = self._image_processor(images=[image], return_tensors="pt")
        single_inputs = {k: v.to(self._device) for k, v in single_inputs.items()}
        with torch.no_grad():
            single_outputs = self._model(**single_inputs)
        single_target = torch.tensor([image.size[::-1]], device=self._device)
        single_raw = self._image_processor.post_process_object_detection(
            single_outputs,
            threshold=pre_threshold,
            target_sizes=single_target,
        )
        return single_raw[0]

    def _post_process_chunk_with_fallback(
        self,
        chunk_pil: List[Image.Image],
        outputs,
        target_sizes,
        pre_threshold: float,
        chunk_start: int,
    ) -> List[Dict]:
        """批处理，失败时逐张重试"""
        try:
            return self._image_processor.post_process_object_detection(
                outputs,
                threshold=pre_threshold,
                target_sizes=target_sizes,
            )
        except Exception as e:
            logger.warning("Layout post_process failed for chunk (retrying image-by-image): %s", e)

        raw_results = []
        for i, img in enumerate(chunk_pil):
            try:
                raw_results.append(self._run_detection_single_image(img, pre_threshold))
            except Exception as e2:
                logger.warning("Layout post_process failed for image %s in chunk: %s", chunk_start + i, e2)
                raw_results.append(self._empty_detection_result())
        return raw_results

    def _expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        img_width: int,
        img_height: int,
        expand_ratio: Optional[float] = None,
        min_expansion: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        """扩展边界框

        Args:
            bbox: (x1, y1, x2, y2) 像素坐标
            img_width: 图像宽度
            img_height: 图像高度
            expand_ratio: 扩展比例，默认使用 self.table_expand_ratio
            min_expansion: 最小扩展像素，默认使用 self.table_min_expansion

        Returns:
            扩展后的边界框
        """
        if expand_ratio is None:
            expand_ratio = self.table_expand_ratio
        if min_expansion is None:
            min_expansion = self.table_min_expansion

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # 计算扩展量
        expand_w = max(int(width * expand_ratio), min_expansion)
        expand_h = max(int(height * expand_ratio), min_expansion)

        # 应用扩展，限制在图像边界内
        new_x1 = max(0, x1 - expand_w)
        new_y1 = max(0, y1 - expand_h)
        new_x2 = min(img_width, x2 + expand_w)
        new_y2 = min(img_height, y2 + expand_h)

        return (new_x1, new_y1, new_x2, new_y2)

    def _pixel_to_norm(self, bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> List[int]:
        """将像素坐标转换为归一化坐标 (0-1000)"""
        x1, y1, x2, y2 = bbox
        return [
            int(float(x1) / img_width * 1000),
            int(float(y1) / img_height * 1000),
            int(float(x2) / img_width * 1000),
            int(float(y2) / img_height * 1000),
        ]

    def _crop_with_polygon(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        polygon: Optional[List[List[float]]] = None,
    ) -> Image.Image:
        """使用边界框和可选的多边形裁剪图像

        Args:
            image: 原始图像
            bbox: (x1, y1, x2, y2) 像素坐标
            polygon: 多边形点列表（归一化坐标，相对于原图）

        Returns:
            裁剪后的图像
        """
        x1, y1, x2, y2 = bbox
        cropped = image.crop((x1, y1, x2, y2))

        if not polygon or len(polygon) < 3:
            return cropped

        # 将归一化多边形转换为裁剪区域内的像素坐标
        img_width, img_height = image.size
        crop_width = x2 - x1
        crop_height = y2 - y1

        polygon_pixels = [
            (
                int(point[0] * crop_width / 1000),
                int(point[1] * crop_height / 1000),
            )
            for point in polygon
        ]

        # 创建掩码并应用
        mask = Image.new("L", cropped.size, 0)
        ImageDraw.Draw(mask).polygon(polygon_pixels, fill=255)
        background = Image.new(cropped.mode, cropped.size, self.fill_color)
        return Image.composite(cropped, background, mask)

    def _merge_non_table_regions(
        self,
        image: Image.Image,
        table_regions: List[TableRegion],
    ) -> NonTableImage:
        """合并非表格区域为整体图像

        策略: 从原图中"挖掉"所有表格区域（填充为白色），保留非表格区域的相对空间位置

        Args:
            image: 原始图像
            table_regions: 表格区域列表

        Returns:
            非表格区域整体图像
        """
        img_width, img_height = image.size

        # 创建副本
        merged = image.copy()
        if merged.mode != "RGB":
            merged = merged.convert("RGB")

        # 填充表格区域为白色
        draw = ImageDraw.Draw(merged)
        for table in table_regions:
            if table.expanded_bbox_pixel:
                x1, y1, x2, y2 = table.expanded_bbox_pixel
            else:
                x1, y1, x2, y2 = table.bbox_pixel
            draw.rectangle([x1, y1, x2, y2], fill=self.fill_color)

        return NonTableImage(
            merged_image=merged,
            original_size=(img_width, img_height),
            cropped_regions_count=len(table_regions),
        )

    def _process_medical_result(
        self,
        image: Image.Image,
        all_regions: List[Dict],
    ) -> MedicalExpenseResult:
        """处理医疗费用清单结果

        Args:
            image: 原始图像
            all_regions: 所有布局检测结果

        Returns:
            MedicalExpenseResult 实例
        """
        result = MedicalExpenseResult()
        result.original_image = image
        result.image_size = image.size
        result.all_regions = all_regions

        img_width, img_height = image.size

        # 分离表格和非表格区域
        table_regions = []
        non_table_count = 0
        table_id_counter = 0

        for item in all_regions:
            label = item["label"]
            bbox_2d = item["bbox_2d"]
            score = item["score"]
            polygon = item.get("polygon")

            # 转换为像素坐标
            x1_norm, y1_norm, x2_norm, y2_norm = bbox_2d
            x1 = int(x1_norm * img_width / 1000)
            y1 = int(y1_norm * img_height / 1000)
            x2 = int(x2_norm * img_width / 1000)
            y2 = int(y2_norm * img_height / 1000)

            # 判断是否为表格
            is_table = any(tab_label.lower() in label.lower() for tab_label in self.table_labels)

            if is_table:
                # 创建表格区域
                table_region = TableRegion(
                    table_id=table_id_counter,
                    bbox_2d=bbox_2d,
                    bbox_pixel=(x1, y1, x2, y2),
                    label=label,
                    score=score,
                    polygon=polygon,
                )

                # 计算扩展后的边界框
                expanded_pixel = self._expand_bbox((x1, y1, x2, y2), img_width, img_height)
                table_region.expanded_bbox_pixel = expanded_pixel
                table_region.expanded_bbox_2d = self._pixel_to_norm(expanded_pixel, img_width, img_height)

                # 裁剪扩展后的表格区域
                table_region.cropped_image = self._crop_with_polygon(
                    image,
                    expanded_pixel,
                    polygon,
                )

                table_regions.append(table_region)
                table_id_counter += 1
            else:
                non_table_count += 1

        result.table_regions = table_regions

        # 处理非表格区域: 合并为整体图像
        if non_table_count > 0 or len(table_regions) > 0:
            result.non_table_image = self._merge_non_table_regions(image, table_regions)

        return result

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        global_start_idx: int = 0,
        use_polygon: bool = False,
        return_medical_result: bool = True,
    ) -> tuple:
        """批量检测布局区域

        Args:
            images: PIL 图像列表
            save_visualization: 是否生成可视化图像
            global_start_idx: 可视化页面编号的起始索引
            use_polygon: 是否使用多边形掩码进行可视化和裁剪
            return_medical_result: 是否返回医疗费用清单特殊结果

        Returns:
            Tuple of (results, vis_images, medical_results) where:
                - results: 标准布局检测结果
                - vis_images: 可视化图像字典
                - medical_results: MedicalExpenseResult 列表（仅当 return_medical_result=True 时）
        """
        if self._model is None:
            raise RuntimeError("Layout detector not started. Call start() first.")
        self._validate_runtime_config()

        num_images = len(images)
        pil_images = [
            img.convert("RGB") if img.mode != "RGB" else img for img in images
        ]
        all_paddle_format_results = []

        for chunk_start in range(0, num_images, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, num_images)
            chunk_pil = pil_images[chunk_start:chunk_end]

            inputs = self._image_processor(images=chunk_pil, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor(
                [img.size[::-1] for img in chunk_pil], device=self._device
            )

            if self.threshold_by_class:
                pre_threshold = min(self.threshold, min(self.threshold_by_class.values()))
            else:
                pre_threshold = self.threshold

            raw_results = self._post_process_chunk_with_fallback(
                chunk_pil, outputs, target_sizes, pre_threshold, chunk_start
            )

            if self.threshold_by_class:
                raw_results = self._apply_per_class_threshold(raw_results)

            img_sizes = [img.size for img in chunk_pil]
            paddle_format_results = apply_layout_postprocess(
                raw_results=raw_results,
                id2label=self.id2label,
                img_sizes=img_sizes,
                layout_nms=self.layout_nms,
                layout_unclip_ratio=self.layout_unclip_ratio,
                layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
            )
            all_paddle_format_results.extend(paddle_format_results)

            if self._device.startswith("cuda") and chunk_end < num_images:
                del inputs, outputs, raw_results
                torch.cuda.empty_cache()

        # 生成可视化
        vis_images: Dict[int, Image.Image] = {}
        if save_visualization:
            for img_idx, img_results in enumerate(all_paddle_format_results):
                vis_img = np.array(pil_images[img_idx])
                vis_images[global_start_idx + img_idx] = draw_layout_boxes(
                    image=vis_img,
                    boxes=img_results,
                    use_polygon=use_polygon,
                )

        # 转换为标准结果格式
        all_results = []
        for img_idx, paddle_results in enumerate(all_paddle_format_results):
            image_width, image_height = pil_images[img_idx].size
            results = []
            valid_index = 0
            for item in paddle_results:
                label = item["label"]
                score = item["score"]
                box = item["coordinate"]
                task_type = None
                for task_item, labels in self.label_task_mapping.items():
                    if isinstance(labels, list) and label in labels:
                        task_type = task_item
                        break
                if task_type is None or task_type == "abandon":
                    continue

                x1, y1, x2, y2 = box
                x1_norm = int(float(x1) / image_width * 1000)
                y1_norm = int(float(y1) / image_height * 1000)
                x2_norm = int(float(x2) / image_width * 1000)
                y2_norm = int(float(y2) / image_height * 1000)

                poly_array = item["polygon_points"]
                polygon = [
                    [
                        int(float(point[0]) / image_width * 1000),
                        int(float(point[1]) / image_height * 1000),
                    ]
                    for point in poly_array
                ]

                results.append({
                    "index": valid_index,
                    "label": label,
                    "score": float(score),
                    "bbox_2d": [x1_norm, y1_norm, x2_norm, y2_norm],
                    "polygon": polygon,
                    "task_type": task_type,
                })
                valid_index += 1
            all_results.append(results)

        # 处理医疗费用清单结果
        medical_results = []
        if return_medical_result:
            for img_idx, (img, regions) in enumerate(zip(pil_images, all_results)):
                medical_result = self._process_medical_result(img, regions)
                medical_results.append(medical_result)

        if return_medical_result:
            return all_results, vis_images, medical_results
        return all_results, vis_images


def create_medical_expense_detector(
    config: "LayoutConfig",
    table_expand_ratio: float = 0.05,
    table_min_expansion: int = 10,
    table_labels: Optional[List[str]] = None,
    fill_color: Tuple[int, int, int] = (255, 255, 255),
) -> MedicalExpenseLayoutDetector:
    """
    工厂函数: 创建医疗费用清单布局检测器

    Args:
        config: LayoutConfig 实例
        table_expand_ratio: 表格扩展比例，默认 0.05 (5%)
        table_min_expansion: 最小扩展像素，默认 10
        table_labels: 表格标签列表，默认 ["table", "表格"]
        fill_color: 非表格区域中表格部分的填充色，默认 (255, 255, 255)

    Returns:
        MedicalExpenseLayoutDetector 实例
    """
    # 设置配置
    config.table_expand_ratio = table_expand_ratio
    config.table_min_expansion = table_min_expansion
    config.table_labels = table_labels or ["table", "表格"]
    config.fill_color = fill_color

    return MedicalExpenseLayoutDetector(config)
