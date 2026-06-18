"""
预处理模型模块

包含各类预处理模型的加载器和推理实现：
- base: 模型加载器抽象基类
- doc_detector: 文档检测模型
- orientation_corrector: 方向矫正模型
- dewarp_corrector: 扭曲矫正模型
"""

from __future__ import annotations

from preprocess.models.base import BaseModelLoader

__all__ = [
    "BaseModelLoader",
]
