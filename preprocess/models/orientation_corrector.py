"""
方向矫正模型加载器

用于检测并矫正文档的旋转方向（0°, 90°, 180°, 270°）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Union

import torch
from PIL import Image
import torchvision.transforms as T

from preprocess.models.base import BaseModelLoader
from preprocess.utils.logging import get_logger

if TYPE_CHECKING:
    from preprocess.config import OrientationCorrectorConfig, ModelConfig

logger = get_logger(__name__)


class OrientationCorrectorLoader(BaseModelLoader):
    """方向矫正模型加载器
    
    模型功能：
    - 输入：文档图像
    - 输出：旋转角度（0°, 90°, 180°, 270°）
    """
    
    def __init__(
        self, 
        config: "ModelConfig", 
        corrector_config: "OrientationCorrectorConfig"
    ):
        super().__init__(config)
        self.corrector_config = corrector_config
        self._transform = None
    
    def load_model(self, model_dir: str, **kwargs: Any) -> None:
        """加载方向矫正模型
        
        Args:
            model_dir: 模型目录路径
            **kwargs: 其他参数
        """
        logger.info(f"加载方向矫正模型: {model_dir}")
        
        # 解析设备
        self._device = self._resolve_device(self.config.device)
        logger.info(f"使用设备: {self._device}")
        
        # TODO: 根据实际模型架构修改
        # 示例：PyTorch 分类模型
        # self._model = torch.load(f"{model_dir}/orientation_model.pth", 
        #                          map_location=self._device)
        # self._model.eval()
        
        # 定义图像预处理
        self._transform = T.Compose([
            T.Resize((self.corrector_config.input_size, 
                     self.corrector_config.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
        ])
        
        self._loaded = True
        logger.info("方向矫正模型加载完成")
    
    def predict(
        self, 
        inputs: Union[Image.Image, List[Image.Image]], 
        **kwargs: Any
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """执行方向检测推理
        
        Args:
            inputs: 单张图像或图像列表
            **kwargs: 其他参数
            
        Returns:
            检测结果，包含旋转角度和置信度
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
            检测结果列表
        """
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
        for output in outputs:
            result = self._postprocess(output)
            results.append(result)
        
        return results
    
    def _dummy_inference(self, batch: torch.Tensor) -> List[Dict[str, Any]]:
        """模拟推理（示例）"""
        batch_size = batch.shape[0]
        results = []
        
        for _ in range(batch_size):
            # 模拟输出：4个类别的概率分布 [0°, 90°, 180°, 270°]
            import torch.nn.functional as F
            logits = torch.randn(1, 4).to(self._device)
            probs = F.softmax(logits, dim=1)[0]
            
            result = {
                "angles": [0, 90, 180, 270],
                "probabilities": probs.cpu().tolist(),
                "predicted_angle": 0,  # 将在后处理中确定
                "confidence": 0.0
            }
            results.append(result)
        
        return results
    
    def _postprocess(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """后处理检测结果
        
        Args:
            output: 模型原始输出
            
        Returns:
            后处理后的结果
        """
        probs = output["probabilities"]
        angles = output["angles"]
        
        # 找到最大概率对应的角度
        max_idx = probs.index(max(probs))
        predicted_angle = angles[max_idx]
        confidence = probs[max_idx]
        
        # 检查是否超过阈值
        if confidence < self.corrector_config.threshold:
            # 置信度太低，认为不需要矫正
            predicted_angle = 0
            confidence = 1.0
        
        return {
            "angle": predicted_angle,
            "confidence": confidence,
            "all_probabilities": dict(zip(angles, probs))
        }
    
    @staticmethod
    def apply_rotation(image: Image.Image, angle: int) -> Image.Image:
        """应用旋转变换
        
        Args:
            image: 输入图像
            angle: 旋转角度（0, 90, 180, 270）
            
        Returns:
            旋转后的图像
        """
        if angle == 0:
            return image
        elif angle == 90:
            return image.transpose(Image.ROTATE_90)
        elif angle == 180:
            return image.transpose(Image.ROTATE_180)
        elif angle == 270:
            return image.transpose(Image.ROTATE_270)
        else:
            logger.warning(f"不支持的旋转角度: {angle}，返回原图")
            return image
