"""
配置管理模块
使用Pydantic进行配置验证
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置"""
    
    # 基础配置
    APP_NAME: str = "GLM-OCR Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS配置
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # 文件上传配置
    UPLOAD_DIR: Path = Path("/tmp/glmocr_uploads")
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "bmp", "pdf"]
    
    # 任务队列配置
    MAX_QUEUE_SIZE: int = 1000
    MAX_CONCURRENT_TASKS: int = 4
    TASK_TIMEOUT: int = 300  # 5分钟
    
    # 速率限制配置
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # 60秒
    
    # GLM-OCR配置
    GLMOCR_CONFIG_PATH: Optional[str] = None
    GLMOCR_MODE: str = "selfhosted"  # maas 或 selfhosted
    GLMOCR_API_KEY: Optional[str] = None
    GLMOCR_API_URL: Optional[str] = None
    GLMOCR_MODEL: Optional[str] = None
    
    # GPU配置
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    settings = Settings()
    
    # 确保上传目录存在
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量
    if settings.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.CUDA_VISIBLE_DEVICES
    
    return settings
