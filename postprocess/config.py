"""后处理服务配置

定义后处理微服务的配置项，支持自定义后处理规则。
"""

from typing import Dict, List, Optional, Any
from pydantic import Field
from pydantic_settings import BaseSettings


class PostprocessConfig(BaseSettings):
    """后处理配置类
    
    支持自定义后处理规则配置，包括文本清理、格式转换、字段过滤等。
    
    Attributes:
        app_name: 应用名称
        app_version: 应用版本
        host: 服务监听地址
        port: 服务监听端口
        debug: 是否启用调试模式
        enable_text_cleaning: 是否启用文本清理
        enable_format_conversion: 是否启用格式转换
        enable_field_filter: 是否启用字段过滤
        custom_rules: 自定义规则配置
        log_level: 日志级别
    """
    
    # 基础配置
    app_name: str = Field(default="Postprocess Service", description="应用名称")
    app_version: str = Field(default="0.1.0", description="应用版本")
    host: str = Field(default="0.0.0.0", description="服务监听地址")
    port: int = Field(default=8002, description="服务监听端口")
    debug: bool = Field(default=False, description="是否启用调试模式")
    
    # 后处理功能开关
    enable_text_cleaning: bool = Field(
        default=True,
        description="是否启用文本清理"
    )
    enable_format_conversion: bool = Field(
        default=True,
        description="是否启用格式转换"
    )
    enable_field_filter: bool = Field(
        default=True,
        description="是否启用字段过滤"
    )
    
    # 文本清理配置
    text_cleaning: Dict[str, Any] = Field(
        default_factory=lambda: {
            "remove_extra_whitespace": True,
            "remove_special_chars": False,
            "normalize_unicode": True,
            "trim_lines": True,
        },
        description="文本清理规则配置"
    )
    
    # 格式转换配置
    format_conversion: Dict[str, Any] = Field(
        default_factory=lambda: {
            "default_format": "json",
            "supported_formats": ["json", "markdown", "html"],
        },
        description="格式转换配置"
    )
    
    # 字段过滤配置
    field_filter: Dict[str, Any] = Field(
        default_factory=lambda: {
            "include_fields": [],
            "exclude_fields": [],
            "rename_fields": {},
        },
        description="字段过滤和重命名配置"
    )
    
    # 自定义规则
    custom_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="自定义后处理规则列表"
    )
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    
    class Config:
        """Pydantic 配置"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 全局配置实例
config = PostprocessConfig()
