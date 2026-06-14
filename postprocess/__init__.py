"""后处理微服务模块

提供 OCR 结果的后处理能力，包括文本清理、格式转换、字段过滤等功能。
"""

from .config import PostprocessConfig
from .processor import ResultProcessor

__version__ = "0.1.0"
__all__ = ["PostprocessConfig", "ResultProcessor"]
