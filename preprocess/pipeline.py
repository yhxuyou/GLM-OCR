"""
预处理流水线

实现串行处理：文档检测 → 方向矫正 → 扭曲矫正
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from PIL import Image

from preprocess.models.doc_detector import DocDetectorLoader
from preprocess.models.orientation_corrector import OrientationCorrectorLoader
from preprocess.models.dewarp_corrector import DewarpCorrectorLoader
from preprocess.utils.logging import get_logger

if TYPE_CHECKING:
    from preprocess.config import PreprocessConfig

logger = get_logger(__name__)


class PreprocessPipeline:
    """预处理流水线
    
    串行执行三个步骤：
    1. 文档检测 - 定位文档区域
    2. 方向矫正 - 矫正旋转角度
    3. 扭曲矫正 - 矫正扭曲变形
    """
    
    def __init__(self, config: "PreprocessConfig"):
        """初始化流水线
        
        Args:
            config: 预处理配置
        """
        self.config = config
        
        # 初始化模型加载器
        self.doc_detector = None
        self.orientation_corrector = None
        self.dewarp_corrector = None
        
        if config.pipeline.enable_doc_detection:
            self.doc_detector = DocDetectorLoader(
                config.model,
                config.doc_detector
            )
        
        if config.pipeline.enable_orientation_correction:
            self.orientation_corrector = OrientationCorrectorLoader(
                config.model,
                config.orientation_corrector
            )
        
        if config.pipeline.enable_dewarp_correction:
            self.dewarp_corrector = DewarpCorrectorLoader(
                config.model,
                config.dewarp_corrector
            )
        
        self._loaded = False
    
    def load_models(self) -> None:
        """加载所有模型"""
        logger.info("开始加载预处理模型...")
        
        if self.doc_detector:
            model_dir = self.config.doc_detector.model_dir
            if model_dir:
                self.doc_detector.load_model(model_dir)
                logger.info("文档检测模型加载完成")
            else:
                logger.warning("未配置文档检测模型路径，跳过加载")
        
        if self.orientation_corrector:
            model_dir = self.config.orientation_corrector.model_dir
            if model_dir:
                self.orientation_corrector.load_model(model_dir)
                logger.info("方向矫正模型加载完成")
            else:
                logger.warning("未配置方向矫正模型路径，跳过加载")
        
        if self.dewarp_corrector:
            model_dir = self.config.dewarp_corrector.model_dir
            if model_dir:
                self.dewarp_corrector.load_model(model_dir)
                logger.info("扭曲矫正模型加载完成")
            else:
                logger.warning("未配置扭曲矫正模型路径，跳过加载")
        
        self._loaded = True
        logger.info("所有预处理模型加载完成")
    
    def warmup(self, sample_image: Optional[Image.Image] = None) -> None:
        """模型预热
        
        Args:
            sample_image: 用于预热的样本图像
        """
        if not self._loaded:
            logger.warning("模型尚未加载，跳过预热")
            return
        
        logger.info("开始模型预热...")
        
        if self.doc_detector and self.doc_detector.is_loaded():
            self.doc_detector.warmup(sample_image)
        
        if self.orientation_corrector and self.orientation_corrector.is_loaded():
            self.orientation_corrector.warmup(sample_image)
        
        if self.dewarp_corrector and self.dewarp_corrector.is_loaded():
            self.dewarp_corrector.warmup(sample_image)
        
        logger.info("模型预热完成")
    
    def process(
        self, 
        image: Image.Image, 
        return_intermediate: Optional[bool] = None
    ) -> Dict[str, Any]:
        """执行预处理流水线
        
        Args:
            image: 输入图像
            return_intermediate: 是否返回中间结果，None 时使用配置值
            
        Returns:
            预处理结果字典
        """
        if not self._loaded:
            raise RuntimeError("模型尚未加载，请先调用 load_models()")
        
        if return_intermediate is None:
            return_intermediate = self.config.pipeline.return_intermediate_results
        
        result = {
            "original_size": image.size,
            "steps": []
        }
        
        current_image = image
        
        # 步骤 1: 文档检测
        if self.doc_detector and self.doc_detector.is_loaded():
            logger.debug("执行文档检测...")
            detection_result = self.doc_detector.predict(current_image)
            
            step_info = {
                "step": "doc_detection",
                "result": detection_result
            }
            result["steps"].append(step_info)
            
            # 如果检测到文档区域，裁剪图像
            if detection_result.get("bbox"):
                bbox = detection_result["bbox"]
                # bbox 格式: [x1, y1, x2, y2]
                current_image = current_image.crop(bbox)
                logger.debug(f"文档检测完成，裁剪区域: {bbox}")
            else:
                logger.debug("未检测到文档区域，使用原图")
        
        # 步骤 2: 方向矫正
        if self.orientation_corrector and self.orientation_corrector.is_loaded():
            logger.debug("执行方向矫正...")
            orientation_result = self.orientation_corrector.predict(current_image)
            
            step_info = {
                "step": "orientation_correction",
                "result": orientation_result
            }
            result["steps"].append(step_info)
            
            # 应用旋转变换
            angle = orientation_result.get("angle", 0)
            if angle != 0:
                current_image = OrientationCorrectorLoader.apply_rotation(
                    current_image, 
                    angle
                )
                logger.debug(f"方向矫正完成，旋转角度: {angle}°")
            else:
                logger.debug("方向矫正完成，无需旋转")
        
        # 步骤 3: 扭曲矫正
        if self.dewarp_corrector and self.dewarp_corrector.is_loaded():
            logger.debug("执行扭曲矫正...")
            dewarp_result = self.dewarp_corrector.predict(current_image)
            
            step_info = {
                "step": "dewarp_correction",
                "result": {
                    "original_size": dewarp_result.get("original_size"),
                    "corrected_size": dewarp_result.get("corrected_size")
                }
            }
            result["steps"].append(step_info)
            
            # 使用矫正后的图像
            current_image = dewarp_result.get("corrected_image", current_image)
            logger.debug("扭曲矫正完成")
        
        # 最终结果
        result["final_image"] = current_image
        result["final_size"] = current_image.size
        
        # 如果不返回中间结果，清理 steps
        if not return_intermediate:
            result.pop("steps")
        
        logger.info(f"预处理完成，最终尺寸: {current_image.size}")
        return result
    
    def process_batch(
        self, 
        images: List[Image.Image], 
        return_intermediate: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """批量执行预处理流水线
        
        Args:
            images: 输入图像列表
            return_intermediate: 是否返回中间结果
            
        Returns:
            预处理结果列表
        """
        results = []
        for idx, image in enumerate(images):
            logger.debug(f"处理图像 {idx + 1}/{len(images)}")
            result = self.process(image, return_intermediate)
            results.append(result)
        
        return results
    
    def unload_models(self) -> None:
        """卸载所有模型"""
        logger.info("开始卸载预处理模型...")
        
        if self.doc_detector:
            self.doc_detector.unload()
        
        if self.orientation_corrector:
            self.orientation_corrector.unload()
        
        if self.dewarp_corrector:
            self.dewarp_corrector.unload()
        
        self._loaded = False
        logger.info("所有预处理模型已卸载")
