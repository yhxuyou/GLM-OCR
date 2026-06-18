"""Pipeline 微服务

端到端文档处理流水线，整合：
  - preprocess: 图像预处理（文档检测、方向矫正、扭曲矫正）
  - glmocr: GLM-OCR 异步识别服务
  - postprocess: 结果后处理
"""

from .config import PipelineConfig, config
from .service import PipelineService
from .client import (
    PreprocessClient,
    GLMOCRClient,
    PostprocessClient,
    ServiceClientError,
)

__all__ = [
    "PipelineConfig",
    "config",
    "PipelineService",
    "PreprocessClient",
    "GLMOCRClient",
    "PostprocessClient",
    "ServiceClientError",
]