"""
预处理微服务

提供文档预处理能力，包括：
- 文档检测：检测图片中的文档区域
- 方向矫正：矫正文档的旋转方向
- 扭曲矫正：矫正文档的扭曲变形

支持独立部署和与 OCR 服务组合部署。
"""

from __future__ import annotations

__version__ = "0.1.0"

# 延迟导入，避免循环依赖
__all__ = [
    "config",
    "models",
    "utils",
]
