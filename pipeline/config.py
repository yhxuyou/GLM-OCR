"""Pipeline 服务配置管理"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServiceConfig:
    """单个服务配置"""
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class PipelineConfig:
    """Pipeline 总配置"""

    # 预处理服务
    preprocess: ServiceConfig = field(default_factory=lambda: ServiceConfig(
        host=os.getenv("PREPROCESS_HOST", "127.0.0.1"),
        port=int(os.getenv("PREPROCESS_PORT", "8001")),
    ))

    # GLM-OCR 异步服务
    glmocr: ServiceConfig = field(default_factory=lambda: ServiceConfig(
        host=os.getenv("GLMOCR_HOST", "127.0.0.1"),
        port=int(os.getenv("GLMOCR_PORT", "8000")),
    ))

    # 后处理服务
    postprocess: ServiceConfig = field(default_factory=lambda: ServiceConfig(
        host=os.getenv("POSTPROCESS_HOST", "127.0.0.1"),
        port=int(os.getenv("POSTPROCESS_PORT", "8002")),
    ))

    # 轮询配置
    poll_interval: float = 1.0  # 秒
    poll_timeout: float = 300.0  # 秒，总超时时间

    # 本服务配置
    host: str = field(default_factory=lambda: os.getenv("PIPELINE_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("PIPELINE_PORT", "8090")))


# 全局单例
config = PipelineConfig()
