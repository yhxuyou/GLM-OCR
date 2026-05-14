"""
医疗费用清单处理 - Pipeline集成适配器
Medical Expense List Pipeline Adapter

用于将医疗费用清单处理器集成到GLM-OCR pipeline中
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image

from glmocr.pipeline import Pipeline
from glmocr.config import load_config
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class MedicalExpensePipelineAdapter:
    """医疗费用清单Pipeline适配器"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        table_labels: Optional[List[str]] = None,
        table_expand_ratio: float = 0.05,
        table_min_expansion: int = 10,
        **kwargs
    ):
        """
        初始化适配器
        
        Args:
            config_path: GLM-OCR配置文件路径
            table_labels: 表格区域标签列表
            table_expand_ratio: 表格扩展比例
            table_min_expansion: 最小扩展像素
            **kwargs: 其他GLM-OCR配置参数
        """
        # 导入医疗费用清单处理器
        try:
            from deploy.medical_expense_processor import (
                MedicalExpenseProcessor,
                MedicalExpenseResult
            )
            self.medical_processor = MedicalExpenseProcessor(
                table_expand_ratio=table_expand_ratio,
                table_min_expansion=table_min_expansion
            )
        except ImportError:
            logger.warning("Medical expense processor not found, using fallback")
            self.medical_processor = None
        
        self.table_labels = table_labels or ["table", "表格"]
        
        # 初始化GLM-OCR Pipeline
        self.config = load_config(config_path, **kwargs)
        self.pipeline = Pipeline(config=self.config.pipeline)
        
        # 存储处理结果
        self._medical_results: Dict[str, MedicalExpenseResult] = {}
    
    def start(self):
        """启动Pipeline"""
        self.pipeline.start()
        logger.info("Medical Expense Pipeline started")
    
    def stop(self):
        """停止Pipeline"""
        self.pipeline.stop()
        logger.info("Medical Expense Pipeline stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def parse(
        self,
        image_source: str,
        return_medical_expense_result: bool = True,
        save_layout_visualization: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        解析医疗费用清单
        
        Args:
            image_source: 图片路径或URL
            return_medical_expense_result: 是否返回医疗费用清单专用结果
            save_layout_visualization: 是否保存布局可视化
            **kwargs: 其他参数
            
        Returns:
            包含标准OCR结果和医疗费用清单专用结果的字典
        """
        # 调用标准Pipeline处理
        request_data = {
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": image_source if image_source.startswith(("http", "file")) else f"file://{image_source}"}
                }]
            }]
        }
        
        results = list(self.pipeline.process(
            request_data,
            save_layout_visualization=save_layout_visualization
        ))
        
        if not results:
            return {"error": "No results from pipeline"}
        
        result = results[0]
        
        # 构建响应
        response = {
            "json_result": result.json_result,
            "markdown_result": result.markdown_result,
            "original_images": result.original_images
        }
        
        # 如果需要医疗费用清单专用结果
        if return_medical_expense_result and self.medical_processor:
            try:
                # 获取原始图像
                image = Image.open(image_source)
                
                # 提取layout结果
                layout_results = self._extract_layout_from_result(result)
                
                # 处理医疗费用清单
                medical_result = self.medical_processor.process(
                    image=image,
                    layout_results=layout_results,
                    table_labels=self.table_labels
                )
                
                # 添加到响应
                response["medical_expense"] = {
                    "table_regions": [
                        {
                            "table_id": t.table_id,
                            "bbox": t.bbox,
                            "bbox_norm": t.bbox_norm,
                            "label": t.label,
                            "score": t.score,
                            "expanded_bbox": t.expanded_bbox,
                            "has_cropped_image": t.cropped_image is not None
                        }
                        for t in medical_result.table_regions
                    ],
                    "non_table_image": {
                        "size": medical_result.non_table_image.merged_image.size if medical_result.non_table_image else None,
                        "original_size": medical_result.image_size
                    },
                    "table_count": len(medical_result.table_regions),
                    "has_non_table_image": medical_result.non_table_image is not None
                }
                
                # 保存医疗处理结果
                self._medical_results[image_source] = medical_result
                
            except Exception as e:
                logger.error(f"Error processing medical expense: {e}")
                response["medical_expense_error"] = str(e)
        
        return response
    
    def _extract_layout_from_result(self, result) -> List[Dict[str, Any]]:
        """
        从Pipeline结果中提取layout信息
        
        这个方法需要根据实际返回的结果结构调整
        """
        layout_results = []
        
        # 尝试从raw_json_result中提取
        if hasattr(result, 'raw_json_result') and result.raw_json_result:
            for page_idx, page_regions in enumerate(result.raw_json_result):
                for region in page_regions:
                    layout_results.append({
                        "label": region.get("label", "text"),
                        "bbox_2d": region.get("bbox_2d", [0, 0, 1000, 1000]),
                        "polygon": region.get("polygon"),
                        "score": region.get("score", 1.0),
                        "content": region.get("content", "")
                    })
        
        return layout_results
    
    def get_medical_result(self, image_source: str) -> Optional[Any]:
        """获取指定图像的医疗费用清单处理结果"""
        return self._medical_results.get(image_source)
    
    def save_results(self, image_source: str, output_dir: str) -> Dict[str, str]:
        """保存处理结果"""
        medical_result = self.get_medical_result(image_source)
        if medical_result and hasattr(medical_result, 'save'):
            return medical_result.save(output_dir)
        return {}


def create_medical_expense_pipeline(
    config_path: Optional[str] = None,
    table_labels: Optional[List[str]] = None,
    table_expand_ratio: float = 0.05,
    table_min_expansion: int = 10,
    **kwargs
) -> MedicalExpensePipelineAdapter:
    """
    工厂函数：创建医疗费用清单Pipeline
    
    Args:
        config_path: 配置文件路径
        table_labels: 表格标签列表
        table_expand_ratio: 表格扩展比例
        table_min_expansion: 最小扩展像素
        **kwargs: 其他配置
        
    Returns:
        MedicalExpensePipelineAdapter实例
    """
    return MedicalExpensePipelineAdapter(
        config_path=config_path,
        table_labels=table_labels,
        table_expand_ratio=table_expand_ratio,
        table_min_expansion=table_min_expansion,
        **kwargs
    )
