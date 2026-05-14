"""
日志模块
"""
import logging
import sys
from typing import Optional
from .config import get_settings


def setup_logger(name: str = "glmocr_service") -> logging.Logger:
    """设置日志器"""
    settings = get_settings()
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 控制台Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # 格式化器
    formatter = logging.Formatter(settings.LOG_FORMAT)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger


# 全局日志器
logger = setup_logger()
