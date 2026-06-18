"""
预处理模型加载基类

定义模型加载、推理、预热等统一接口
支持 GPU/CPU 设备选择
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from PIL import Image

from preprocess.utils.logging import get_logger

if TYPE_CHECKING:
    from preprocess.config import ModelConfig

logger = get_logger(__name__)


class BaseModelLoader(ABC):
    """预处理模型加载器抽象基类

    定义统一的模型加载、推理、预热接口。
    子类需要实现 load_model() 和 predict() 方法。

    Attributes:
        config: 模型全局配置
        _model: 加载的模型实例
        _device: 运行设备（cpu / cuda / cuda:N）
        _loaded: 模型是否已加载
    """

    def __init__(self, config: "ModelConfig"):
        """初始化模型加载器

        Args:
            config: ModelConfig 实例，包含设备和 CUDA 配置
        """
        self.config = config
        self._model: Any = None
        self._device: Optional[str] = None
        self._loaded: bool = False

    def _resolve_device(self, config_device: Optional[str] = None) -> str:
        """解析运行设备

        设备选择优先级：
        1. 显式指定的 config_device（"cpu", "cuda", "cuda:N"）
        2. 自动选择：CUDA 可用时使用 cuda:{cuda_visible_devices}，否则 CPU

        Args:
            config_device: 显式指定的设备字符串，可为 None

        Returns:
            解析后的设备字符串
        """
        if config_device is not None:
            return config_device

        if torch.cuda.is_available() and self.config.cuda_visible_devices:
            return f"cuda:{self.config.cuda_visible_devices}"

        return "cpu"

    def load_model(self, model_dir: str, **kwargs: Any) -> None:
        """加载模型

        子类必须实现此方法。加载完成后应将模型赋值到 self._model，
        并将 self._device 设置为实际使用的设备，self._loaded 设为 True。

        Args:
            model_dir: 模型目录路径（本地路径或 Hugging Face 模型 ID）
            **kwargs: 子类特定的加载参数
        """
        raise NotImplementedError("子类必须实现 load_model() 方法")

    @abstractmethod
    def predict(
        self, inputs: Union[Image.Image, List[Image.Image]], **kwargs: Any
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """执行推理

        Args:
            inputs: 单张 PIL 图像或图像列表
            **kwargs: 子类特定的推理参数

        Returns:
            单张图像的推理结果字典，或图像列表的推理结果字典列表
        """

    def warmup(self, sample_input: Optional[Image.Image] = None) -> None:
        """模型预热

        使用一个样本输入执行一次前向推理，确保模型已加载到设备并就绪。
        子类可覆盖此方法以实现自定义预热逻辑。

        Args:
            sample_input: 用于预热的样本图像。若为 None，
                          则使用一个随机生成的 dummy 图像。
        """
        if not self._loaded:
            logger.warning("模型尚未加载，跳过预热")
            return

        if sample_input is None:
            # 生成一个 dummy 图像用于预热
            sample_input = Image.new("RGB", (224, 224), color=(128, 128, 128))
            logger.debug("使用 dummy 图像进行预热")

        logger.debug("开始模型预热...")
        try:
            self.predict(sample_input)
            logger.debug("模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败（非致命）: {e}")

    def is_loaded(self) -> bool:
        """检查模型是否已加载

        Returns:
            模型是否已加载
        """
        return self._loaded

    @property
    def device(self) -> Optional[str]:
        """获取当前运行设备

        Returns:
            设备字符串，未加载时返回 None
        """
        return self._device

    @property
    def model(self) -> Any:
        """获取模型实例

        Returns:
            模型实例，未加载时返回 None
        """
        return self._model

    def unload(self) -> None:
        """卸载模型，释放资源

        将模型从设备移除并清理引用。子类可覆盖此方法以执行额外的清理操作。
        """
        if self._model is not None:
            try:
                del self._model
            except Exception:
                pass
            self._model = None
        self._loaded = False
        logger.debug("模型已卸载")

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(device={self._device}, loaded={self._loaded})"
        )
