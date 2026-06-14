"""
预处理微服务日志工具

提供统一的日志系统：
- INFO 级别：面向用户的进度消息
- DEBUG 级别：详细调试信息和性能分析
"""

from __future__ import annotations

import logging
from typing import Optional

# 包级日志器名称
_PACKAGE_LOGGER_NAME = "preprocess"
_configured = False
_configured_source: Optional[str] = None

# 默认格式
_INFO_FORMAT = "%(message)s"
_DEBUG_FORMAT = "[%(levelname)s] %(name)s: %(message)s"


def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    *,
    _source: str = "explicit",
) -> None:
    """配置 preprocess 包的日志系统

    Args:
        level: 日志级别（"DEBUG", "INFO", "WARNING", "ERROR"）
        format_string: 自定义格式字符串。若为 None，则根据级别使用默认格式
    """
    global _configured, _configured_source

    # 获取或创建包日志器
    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)

    # 解析级别
    level_value = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level_value)

    # 移除已有的处理器
    logger.handlers.clear()

    # 创建处理器
    handler = logging.StreamHandler()
    handler.setLevel(level_value)

    # 设置格式
    if format_string is None:
        format_string = _DEBUG_FORMAT if level_value == logging.DEBUG else _INFO_FORMAT

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 阻止向根日志器传播
    logger.propagate = False

    _configured = True
    _configured_source = _source


def get_logger(name: str) -> logging.Logger:
    """获取指定模块名称的日志器

    Args:
        name: 模块名称（通常是 __name__）

    Returns:
        日志器实例

    示例::

        logger = get_logger(__name__)
        logger.info("处理开始")
    """
    global _configured

    # 确保包日志已配置
    if not _configured:
        configure_logging(_source="auto")

    # 返回包命名空间下的子日志器
    if name.startswith(_PACKAGE_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_PACKAGE_LOGGER_NAME}.{name}")


def set_log_level(level: str) -> None:
    """设置 preprocess 包的日志级别

    Args:
        level: 日志级别（"DEBUG", "INFO", "WARNING", "ERROR"）
    """
    configure_logging(level=level)


def ensure_logging_configured(
    level: str = "INFO", format_string: Optional[str] = None
) -> None:
    """确保 preprocess 日志已从外部配置中配置

    如果日志只是自动配置的（get_logger() 触发的隐式默认值），
    这将使用提供的级别/格式重新配置它。

    这避免了常见的陷阱：导入模块会创建日志器并将日志锁定
    在默认的 INFO 级别，即使配置文件指定了 DEBUG。
    """
    global _configured, _configured_source

    if (not _configured) or (_configured_source == "auto"):
        configure_logging(level=level, format_string=format_string, _source="explicit")
