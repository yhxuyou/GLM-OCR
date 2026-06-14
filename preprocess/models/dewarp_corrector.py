"""
扭曲矫正模型加载器

用于矫正文档的扭曲变形（如弯曲、褶皱等）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np

from preprocess.models.base import BaseModelLoader
from preprocess.utils.logging import get_logger

if TYPE_CHECKING:
    from preprocess.config import DewarpCorrectorConfig, ModelConfig

logger = get_logger(__name__)


class DewarpCorrectorLoader(BaseModelLoader):
    """扭曲矫正模型加载器
    
    模型功能：
    - 输入：扭曲的文档图像
    - 输出：矫正后的文档图像
    """
    
    def __init__(
        self, 
        config: "ModelConfig", 
        corrector_config: "DewarpCorrectorConfig"
    ):
        super().__init__(config)
        self.corrector_config = corrector_config
        self._transform = None
    
    def load_model(self, model_dir: str, **kwargs: Any) -> None:
        """加载扭曲矫正模型
        
        Args:
            model_dir: 模型目录路径
            **kwargs: 其他参数
        """
        logger.info(f"加载扭曲矫正模型: {model_dir}")
        
        # 解析设备
        self._device = self._resolve_device(self.config.device)
        logger.info(f"使用设备: {self._device}")
        
        # TODO: 根据实际模型架构修改
        # 示例：U-Net 类型的图像到图像模型
        # self._model = torch.load(f"{model_dir}/dewarp_model.pth", 
        #                          map_location=self._device)
        # self._model.eval()
        
        # 定义图像预处理
        self._transform = T.Compose([
            T.Resize((self.corrector_config.input_size, 
                     self.corrector_config.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], 
                       std=[0.5, 0.5, 0.5]),
        ])
        
        self._loaded = True
        logger.info("扭曲矫正模型加载完成")
    
    def predict(
        self, 
        inputs: Union[Image.Image, List[Image.Image]], 
        **kwargs: Any
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """执行扭曲矫正推理
        
        Args:
            inputs: 单张图像或图像列表
            **kwargs: 其他参数
            
        Returns:
            矫正结果，包含矫正后的图像
        """
        if not self._loaded:
            raise RuntimeError("模型尚未加载，请先调用 load_model()")
        
        # 统一处理为列表
        if isinstance(inputs, Image.Image):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        results = []
        
        # 批处理
        batch_size = self.corrector_config.batch_size
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_result = self._predict_batch(batch, **kwargs)
            results.extend(batch_result)
        
        return results[0] if single_input else results
    
    def _predict_batch(
        self, 
        images: List[Image.Image], 
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """批量推理
        
        Args:
            images: 图像列表
            **kwargs: 其他参数
            
        Returns:
            矫正结果列表
        """
        # 保存原始尺寸
        original_sizes = [img.size for img in images]
        
        # 预处理
        batch_tensors = []
        for img in images:
            img_tensor = self._transform(img)
            batch_tensors.append(img_tensor)
        
        # 堆叠为 batch
        batch = torch.stack(batch_tensors).to(self._device)
        
        # 推理
        with torch.no_grad():
            # TODO: 根据实际模型修改推理逻辑
            # outputs = self._model(batch)
            
            # 模拟输出
            outputs = self._dummy_inference(batch)
        
        # 后处理
        results = []
        for idx, output in enumerate(outputs):
            orig_size = original_sizes[idx]
            result = self._postprocess(output, orig_size)
            results.append(result)
        
        return results
    
    def _dummy_inference(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """模拟推理（示例）
        
        实际模型应该输出矫正后的图像张量
        """
        # 模拟：返回输入本身（实际应该是矫正后的图像）
        return [batch]
    
    def _postprocess(
        self, 
        output: torch.Tensor, 
        original_size: tuple
    ) -> Dict[str, Any]:
        """后处理矫正结果
        
        Args:
            output: 模型输出张量
            original_size: 原始图像尺寸 (width, height)
            
        Returns:
            后处理后的结果
        """
        # 反归一化
        # 输入时使用了 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        # 所以反归一化：x = x * 0.5 + 0.5
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0, 1)
        
        # 转换为 numpy 数组
        output_np = output.cpu().permute(1, 2, 0).numpy()
        output_np = (output_np * 255).astype(np.uint8)
        
        # 转换为 PIL 图像
        corrected_image = Image.fromarray(output_np)
        
        # 恢复到原始尺寸
        if corrected_image.size != original_size:
            # 根据配置选择插值方法
            if self.corrector_config.interpolation == "bilinear":
                resample = Image.BILINEAR
            elif self.corrector_config.interpolation == "bicubic":
                resample = Image.BICUBIC
            else:
                resample = Image.NEAREST
            
            corrected_image = corrected_image.resize(original_size, resample)
        
        return {
            "corrected_image": corrected_image,
            "original_size": original_size,
            "corrected_size": corrected_image.size
        }
