"""
医疗费用清单处理器
Medical Expense List Processor

定制化处理医疗费用清单，支持：
1. 表格区域识别和扩展裁剪
2. 非表格区域整体合并
3. 结构化输出

作者：GLM-OCR
版本：1.0.0
"""

import os
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TableRegion:
    """表格区域数据结构"""
    table_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 像素坐标
    bbox_norm: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 归一化坐标 (0-1000)
    label: str
    score: float
    polygon: Optional[List[List[float]]] = None
    expanded_bbox: Optional[Tuple[int, int, int, int]] = None  # 扩展后的bbox
    cropped_image: Optional[Image.Image] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "table_id": self.table_id,
            "bbox": self.bbox,
            "bbox_norm": self.bbox_norm,
            "label": self.label,
            "score": self.score,
            "polygon": self.polygon,
            "expanded_bbox": self.expanded_bbox,
            "has_cropped_image": self.cropped_image is not None
        }


@dataclass
class NonTableImage:
    """非表格区域整体图像"""
    merged_image: Image.Image
    original_size: Tuple[int, int]
    cropped_regions_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "original_size": self.original_size,
            "cropped_regions_count": self.cropped_regions_count,
            "image_size": self.merged_image.size
        }


@dataclass
class MedicalExpenseResult:
    """医疗费用清单处理结果"""
    table_regions: List[TableRegion] = field(default_factory=list)
    non_table_image: Optional[NonTableImage] = None
    original_image: Optional[Image.Image] = None
    image_size: Tuple[int, int] = (0, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "table_count": len(self.table_regions),
            "table_regions": [t.to_dict() for t in self.table_regions],
            "has_non_table_image": self.non_table_image is not None,
            "non_table_image_info": self.non_table_image.to_dict() if self.non_table_image else None,
            "image_size": self.image_size
        }
    
    def save(self, output_dir: str) -> Dict[str, str]:
        """保存所有结果到指定目录
        
        Args:
            output_dir: 输出目录路径
            
        Returns:
            保存的文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 保存表格区域
        for i, table in enumerate(self.table_regions):
            if table.cropped_image:
                table_path = output_path / f"table_{i:03d}.png"
                table.cropped_image.save(table_path, format="PNG")
                saved_files[f"table_{i:03d}"] = str(table_path)
        
        # 保存非表格区域整体图
        if self.non_table_image and self.non_table_image.merged_image:
            non_table_path = output_path / "non_table_regions.png"
            self.non_table_image.merged_image.save(non_table_path, format="PNG")
            saved_files["non_table_regions"] = str(non_table_path)
        
        # 保存原始图像
        if self.original_image:
            original_path = output_path / "original.png"
            self.original_image.save(original_path, format="PNG")
            saved_files["original"] = str(original_path)
        
        return saved_files


class MedicalExpenseProcessor:
    """医疗费用清单处理器"""
    
    TABLE_EXPAND_RATIO = 0.05  # 扩展5%
    TABLE_MIN_EXPANSION = 10  # 最小扩展10像素
    
    def __init__(
        self,
        table_expand_ratio: float = 0.05,
        table_min_expansion: int = 10,
        save_intermediate: bool = False
    ):
        """
        初始化医疗费用清单处理器
        
        Args:
            table_expand_ratio: 表格区域扩展比例 (0.05 = 5%)
            table_min_expansion: 最小扩展像素数
            save_intermediate: 是否保存中间结果
        """
        self.table_expand_ratio = table_expand_ratio
        self.table_min_expansion = table_min_expansion
        self.save_intermediate = save_intermediate
    
    def process(
        self,
        image: Image.Image,
        layout_results: List[Dict[str, Any]],
        table_labels: Optional[List[str]] = None
    ) -> MedicalExpenseResult:
        """
        处理医疗费用清单
        
        Args:
            image: PIL Image 原始图像
            layout_results: 布局检测结果列表，每个元素包含：
                - label: 区域标签
                - bbox_2d: 归一化坐标 [x1, y1, x2, y2] (0-1000)
                - polygon: 多边形点列表 (可选)
                - score: 置信度
            table_labels: 表格区域标签列表，默认为 ["table", "表格"]
            
        Returns:
            MedicalExpenseResult: 处理结果
        """
        if table_labels is None:
            table_labels = ["table", "表格"]
        
        result = MedicalExpenseResult()
        result.original_image = image
        result.image_size = image.size
        
        img_width, img_height = image.size
        
        # 分离表格和非表格区域
        table_regions = []
        non_table_indices = []
        
        for idx, region in enumerate(layout_results):
            label = region.get("label", "").lower()
            
            # 检查是否为表格区域
            is_table = any(table_label.lower() in label for table_label in table_labels)
            
            if is_table:
                # 创建表格区域对象
                bbox_norm = region.get("bbox_2d", [0, 0, 0, 0])
                score = region.get("score", 1.0)
                polygon = region.get("polygon")
                
                # 转换为像素坐标
                x1_norm, y1_norm, x2_norm, y2_norm = bbox_norm
                x1 = int(x1_norm * img_width / 1000)
                y1 = int(y1_norm * img_height / 1000)
                x2 = int(x2_norm * img_width / 1000)
                y2 = int(y2_norm * img_height / 1000)
                
                table_region = TableRegion(
                    table_id=idx,
                    bbox=(x1, y1, x2, y2),
                    bbox_norm=bbox_norm,
                    label=region.get("label", "table"),
                    score=score,
                    polygon=polygon
                )
                
                # 计算扩展后的bbox
                expanded_bbox = self._expand_bbox(
                    (x1, y1, x2, y2),
                    img_width,
                    img_height,
                    self.table_expand_ratio,
                    self.table_min_expansion
                )
                table_region.expanded_bbox = expanded_bbox
                
                # 裁剪扩展后的表格区域
                cropped = self._crop_with_polygon(
                    image,
                    expanded_bbox,
                    polygon
                )
                table_region.cropped_image = cropped
                
                table_regions.append(table_region)
            else:
                non_table_indices.append(idx)
        
        result.table_regions = table_regions
        
        # 处理非表格区域：合并为整体图像
        if non_table_indices:
            non_table_image = self._merge_non_table_regions(
                image,
                layout_results,
                non_table_indices,
                table_regions
            )
            result.non_table_image = non_table_image
        
        logger.info(
            f"Medical expense processing complete: "
            f"{len(table_regions)} tables, "
            f"{len(non_table_indices)} non-table regions"
        )
        
        return result
    
    def _expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        img_width: int,
        img_height: int,
        expand_ratio: float,
        min_expansion: int
    ) -> Tuple[int, int, int, int]:
        """
        扩展边界框
        
        Args:
            bbox: (x1, y1, x2, y2)
            img_width: 图像宽度
            img_height: 图像高度
            expand_ratio: 扩展比例
            min_expansion: 最小扩展像素
            
        Returns:
            扩展后的 (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 计算扩展量
        expand_x = max(int(width * expand_ratio), min_expansion)
        expand_y = max(int(height * expand_ratio), min_expansion)
        
        # 应用扩展（确保不超过图像边界）
        new_x1 = max(0, x1 - expand_x)
        new_y1 = max(0, y1 - expand_y)
        new_x2 = min(img_width, x2 + expand_x)
        new_y2 = min(img_height, y2 + expand_y)
        
        return (new_x1, new_y1, new_x2, new_y2)
    
    def _crop_with_polygon(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        polygon: Optional[List[List[float]]],
        fill_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        使用bbox和多边形裁剪图像
        
        Args:
            image: 原始图像
            bbox: 裁剪边界框 (像素坐标)
            polygon: 多边形点列表 (归一化坐标 0-1000，相对于原图)
            fill_color: 填充颜色
            
        Returns:
            裁剪后的图像
        """
        x1, y1, x2, y2 = bbox
        cropped = image.crop((x1, y1, x2, y2))
        
        # 如果没有多边形，直接返回裁剪结果
        if not polygon or len(polygon) < 3:
            return cropped
        
        # 计算多边形在裁剪区域内的坐标
        # polygon坐标是相对于原图的归一化坐标(0-1000)
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # 将归一化坐标转换为裁剪区域内的像素坐标
        # 关键：需要减去bbox起点并缩放，而不是直接缩放
        polygon_pixels = [
            (
                int((point[0] - x1) * crop_width / 1000),
                int((point[1] - y1) * crop_height / 1000)
            )
            for point in polygon
        ]
        
        # 创建多边形掩码
        mask = Image.new("L", cropped.size, 0)
        ImageDraw.Draw(mask).polygon(polygon_pixels, fill=255)
        
        # 应用掩码
        background = Image.new(cropped.mode, cropped.size, fill_color)
        return Image.composite(cropped, background, mask)
    
    def _merge_non_table_regions(
        self,
        image: Image.Image,
        layout_results: List[Dict[str, Any]],
        non_table_indices: List[int],
        table_regions: List[TableRegion]
    ) -> NonTableImage:
        """
        合并非表格区域为整体图像
        
        策略：从原图中"挖掉"所有表格区域，保留非表格区域的相对空间位置
        
        Args:
            image: 原始图像
            layout_results: 所有布局检测结果
            non_table_indices: 非表格区域的索引列表
            table_regions: 表格区域列表
            
        Returns:
            合并后的非表格区域图像
        """
        img_width, img_height = image.size
        
        # 方案：从原图创建副本，将表格区域填充为白色
        merged = image.copy()
        
        # 确保图像是RGB模式
        if merged.mode != "RGB":
            merged = merged.convert("RGB")
        
        # 绘制所有表格区域为白色（覆盖）
        from PIL import ImageDraw
        draw = ImageDraw.Draw(merged)
        
        for table in table_regions:
            x1, y1, x2, y2 = table.bbox
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
        
        return NonTableImage(
            merged_image=merged,
            original_size=(img_width, img_height),
            cropped_regions_count=len(non_table_indices)
        )
    
    def _extract_non_table_regions_individually(
        self,
        image: Image.Image,
        layout_results: List[Dict[str, Any]],
        non_table_indices: List[int]
    ) -> List[Image.Image]:
        """
        单独提取每个非表格区域（备用方法）
        
        Args:
            image: 原始图像
            layout_results: 所有布局检测结果
            non_table_indices: 非表格区域的索引列表
            
        Returns:
            非表格区域的裁剪图像列表
        """
        img_width, img_height = image.size
        cropped_images = []
        
        for idx in non_table_indices:
            region = layout_results[idx]
            bbox_norm = region.get("bbox_2d", [0, 0, 0, 0])
            polygon = region.get("polygon")
            
            # 转换为像素坐标
            x1_norm, y1_norm, x2_norm, y2_norm = bbox_norm
            x1 = int(x1_norm * img_width / 1000)
            y1 = int(y1_norm * img_height / 1000)
            x2 = int(x2_norm * img_width / 1000)
            y2 = int(y2_norm * img_height / 1000)
            
            cropped = self._crop_with_polygon(
                image,
                (x1, y1, x2, y2),
                polygon
            )
            cropped_images.append(cropped)
        
        return cropped_images


def create_medical_expense_processor(
    table_expand_ratio: float = 0.05,
    table_min_expansion: int = 10,
    save_intermediate: bool = False
) -> MedicalExpenseProcessor:
    """
    工厂函数：创建医疗费用清单处理器
    
    Args:
        table_expand_ratio: 表格区域扩展比例
        table_min_expansion: 最小扩展像素数
        save_intermediate: 是否保存中间结果
        
    Returns:
        MedicalExpenseProcessor实例
    """
    return MedicalExpenseProcessor(
        table_expand_ratio=table_expand_ratio,
        table_min_expansion=table_min_expansion,
        save_intermediate=save_intermediate
    )
