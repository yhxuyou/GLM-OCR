"""
预处理微服务配置模块

包含文档检测、方向矫正、扭曲矫正的配置类
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator

# 环境变量前缀
ENV_PREFIX = "PREPROCESS_"


def _find_dotenv(start: Optional[Path] = None) -> Optional[Path]:
    """从 start 目录开始向上查找 .env 文件

    Args:
        start: 起始目录，默认为当前目录

    Returns:
        .env 文件路径，未找到返回 None
    """
    cur = (start or Path.cwd()).resolve()
    for directory in (cur, *cur.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


# 环境变量映射：环境变量名（不含前缀）-> 配置路径
_ENV_MAP: Dict[str, str] = {
    # 服务配置
    "HOST": "server.host",
    "PORT": "server.port",
    "DEBUG": "server.debug",
    # 日志配置
    "LOG_LEVEL": "logging.level",
    # 设备配置
    "DEVICE": "model.device",
    "CUDA_VISIBLE_DEVICES": "model.cuda_visible_devices",
    # 文档检测配置
    "DOC_DETECTOR_MODEL_DIR": "doc_detector.model_dir",
    "DOC_DETECTOR_THRESHOLD": "doc_detector.threshold",
    "DOC_DETECTOR_BATCH_SIZE": "doc_detector.batch_size",
    # 方向矫正配置
    "ORIENTATION_MODEL_DIR": "orientation_corrector.model_dir",
    "ORIENTATION_THRESHOLD": "orientation_corrector.threshold",
    # 扭曲矫正配置
    "DEWARP_MODEL_DIR": "dewarp_corrector.model_dir",
    "DEWARP_BATCH_SIZE": "dewarp_corrector.batch_size",
}


class _BaseConfig(BaseModel):
    """配置基类"""
    model_config = ConfigDict(extra="allow")


class ServerConfig(_BaseConfig):
    """服务配置"""
    host: str = "0.0.0.0"
    port: int = 5003
    debug: bool = False


class LoggingConfig(_BaseConfig):
    """日志配置"""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    format: Optional[str] = None


class ModelConfig(_BaseConfig):
    """模型全局配置"""
    # 设备选择：cpu, cuda, cuda:0, cuda:1 等
    device: Optional[str] = None
    # CUDA 可见设备
    cuda_visible_devices: str = "0"

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: Optional[str]) -> Optional[str]:
        """验证设备字符串

        允许的值：
        - None / null（根据 CUDA 可用性自动选择）
        - "cpu"
        - "cuda"
        - "cuda:<int>"（如 "cuda:0", "cuda:1"）
        """
        if value is None:
            return value
        v = value.strip()
        if v == "":
            return None
        if v == "cpu" or v == "cuda":
            return v
        if v.startswith("cuda:"):
            index_part = v[5:]
            if index_part.isdigit():
                return v
        raise ValueError(
            "无效的设备值。期望: None, 'cpu', 'cuda', 或 'cuda:<int>'（如 'cuda:0'）"
        )


class DocDetectorConfig(_BaseConfig):
    """文档检测配置

    用于检测图片中的文档区域，返回文档边界框
    """
    # 模型目录（本地路径或 Hugging Face 模型 ID）
    model_dir: Optional[str] = None
    # 检测阈值
    threshold: float = 0.5
    # 批处理大小
    batch_size: int = 1
    # 输入图像最大尺寸
    max_input_size: int = 1024
    # 是否使用 GPU
    use_gpu: bool = True


class OrientationCorrectorConfig(_BaseConfig):
    """方向矫正配置

    用于检测并矫正文档的旋转方向（0°, 90°, 180°, 270°）
    """
    # 模型目录
    model_dir: Optional[str] = None
    # 分类阈值
    threshold: float = 0.7
    # 批处理大小
    batch_size: int = 1
    # 输入图像尺寸
    input_size: int = 224
    # 是否启用
    enabled: bool = True


class DewarpCorrectorConfig(_BaseConfig):
    """扭曲矫正配置

    用于矫正文档的扭曲变形（如弯曲、褶皱等）
    """
    # 模型目录
    model_dir: Optional[str] = None
    # 批处理大小
    batch_size: int = 1
    # 输入图像尺寸
    input_size: int = 512
    # 是否启用
    enabled: bool = True
    # 插值方法：bilinear, bicubic, nearest
    interpolation: str = "bilinear"


class PipelineConfig(_BaseConfig):
    """预处理流水线配置"""
    # 是否启用文档检测
    enable_doc_detection: bool = True
    # 是否启用方向矫正
    enable_orientation_correction: bool = True
    # 是否启用扭曲矫正
    enable_dewarp_correction: bool = True
    # 是否返回中间结果
    return_intermediate_results: bool = False


class PreprocessConfig(_BaseConfig):
    """预处理微服务主配置"""
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    doc_detector: DocDetectorConfig = Field(default_factory=DocDetectorConfig)
    orientation_corrector: OrientationCorrectorConfig = Field(
        default_factory=OrientationCorrectorConfig
    )
    dewarp_corrector: DewarpCorrectorConfig = Field(
        default_factory=DewarpCorrectorConfig
    )
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def default_path(cls) -> str:
        """获取默认配置文件路径"""
        return str(Path(__file__).with_name("config.yaml"))

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]] = None) -> "PreprocessConfig":
        """从 YAML 文件加载配置

        Args:
            path: YAML 文件路径，默认使用同目录下的 config.yaml

        Returns:
            PreprocessConfig 实例
        """
        path = Path(path or cls.default_path())
        if not path.exists():
            raise FileNotFoundError(f"配置文件未找到: {path}")

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    @classmethod
    def from_env(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> "PreprocessConfig":
        """构建配置，支持分层优先级（从高到低）：

        1. 关键字参数覆盖
        2. 环境变量（PREPROCESS_* 前缀）/ .env 文件
        3. YAML 配置文件
        4. 内置默认值

        Args:
            config_path: YAML 配置文件路径
            **overrides: 关键字参数覆盖

        接受的关键字参数：
            - host: 服务主机
            - port: 服务端口
            - device: 设备选择
            - log_level: 日志级别
            - env_file: .env 文件路径

        Returns:
            PreprocessConfig 实例
        """
        # 1. YAML 基线（最低优先级）
        yaml_path = Path(config_path or cls.default_path())
        if yaml_path.exists():
            data: Dict[str, Any] = (
                yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            )
        else:
            if config_path is not None:
                raise FileNotFoundError(f"配置文件未找到: {yaml_path}")
            data = {}

        # 2. 环境变量覆盖
        env_file = overrides.pop("env_file", None)
        env_data = _collect_env_overrides(env_file=env_file)
        if env_data:
            _deep_merge(data, env_data)

        # 3. 关键字参数覆盖
        _KW_MAP = {
            "host": "server.host",
            "port": "server.port",
            "debug": "server.debug",
            "device": "model.device",
            "cuda_visible_devices": "model.cuda_visible_devices",
            "log_level": "logging.level",
        }

        for kw, dotted in _KW_MAP.items():
            if kw in overrides and overrides[kw] is not None:
                raw = overrides[kw]
                _set_nested(data, dotted, _coerce_env_value(dotted, str(raw)))

        return cls.model_validate(data)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()


def _set_nested(data: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """使用点分隔路径设置嵌套字典的值"""
    keys = dotted_path.split(".")
    d = data
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _coerce_env_value(dotted_path: str, raw: str) -> Any:
    """将环境变量字符串转换为预期的 Python 类型"""
    # 布尔字段
    if dotted_path.endswith((".debug", ".enabled", ".use_gpu", ".return_intermediate_results",
                             ".enable_doc_detection", ".enable_orientation_correction",
                             ".enable_dewarp_correction")):
        return raw.strip().lower() in ("true", "1", "yes", "on")
    # 整数字段
    if dotted_path.endswith((".port", ".batch_size", ".max_input_size", ".input_size")):
        return int(raw)
    # 浮点数字段
    if dotted_path.endswith((".threshold",)):
        return float(raw)
    return raw


def _collect_env_overrides(
    env_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """从 .env 文件和环境变量读取配置

    Args:
        env_file: 显式指定 .env 文件路径

    Returns:
        嵌套的配置字典
    """
    # 1. 加载 .env 文件（不修改 os.environ）
    if env_file is not None:
        dotenv_path = Path(env_file)
        if not dotenv_path.is_file():
            raise FileNotFoundError(f".env 文件未找到: {dotenv_path}")
    else:
        dotenv_path = _find_dotenv()
    dotenv_vars: Dict[str, Optional[str]] = (
        dotenv_values(dotenv_path) if dotenv_path else {}
    )

    # 2. 合并：真实环境变量 > .env 文件
    merged: Dict[str, str] = {}
    for env_suffix in _ENV_MAP:
        full_key = f"{ENV_PREFIX}{env_suffix}"
        # 真实环境变量优先
        val = os.environ.get(full_key)
        if val is None:
            val = dotenv_vars.get(full_key)  # type: ignore[assignment]
        if val is not None:
            merged[env_suffix] = val

    # 3. 构建嵌套配置字典
    overrides: Dict[str, Any] = {}
    for env_suffix, raw in merged.items():
        dotted_path = _ENV_MAP[env_suffix]
        _set_nested(overrides, dotted_path, _coerce_env_value(dotted_path, raw))
    return overrides


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并 override 到 base（修改 base）"""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    path: Optional[Union[str, Path]] = None,
    **overrides: Any,
) -> PreprocessConfig:
    """加载配置，优先级：关键字参数 > 环境变量 > YAML > 默认值

    Args:
        path: YAML 配置文件路径
        **overrides: 关键字参数覆盖

    Returns:
        PreprocessConfig 实例
    """
    return PreprocessConfig.from_env(config_path=path, **overrides)
