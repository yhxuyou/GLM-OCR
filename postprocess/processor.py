"""后处理处理器

实现常见的后处理逻辑，包括文本清理、格式转换、字段过滤和自定义规则引擎。
支持插件式扩展。
"""

import re
import json
import html
from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from copy import deepcopy

from .config import PostprocessConfig


class BaseProcessor(ABC):
    """处理器基类
    
    所有后处理插件都应继承此基类并实现 process 方法。
    """
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """处理数据
        
        Args:
            data: 待处理的数据
            **kwargs: 额外的处理参数
            
        Returns:
            处理后的数据
        """
        pass


class TextCleaningProcessor(BaseProcessor):
    """文本清理处理器
    
    去除多余空白、特殊字符，规范化 Unicode 等。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化文本清理处理器
        
        Args:
            config: 文本清理配置
        """
        self.config = config
    
    def process(self, data: Any, **kwargs) -> Any:
        """清理文本数据
        
        Args:
            data: 待清理的数据，可以是字符串或包含字符串的字典/列表
            **kwargs: 额外参数
            
        Returns:
            清理后的数据
        """
        if isinstance(data, str):
            return self._clean_text(data)
        elif isinstance(data, dict):
            return {k: self.process(v, **kwargs) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.process(item, **kwargs) for item in data]
        else:
            return data
    
    def _clean_text(self, text: str) -> str:
        """清理单个文本字符串
        
        Args:
            text: 待清理的文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return text
        
        # 去除多余空白
        if self.config.get("remove_extra_whitespace", True):
            # 将多个连续空格替换为单个空格
            text = re.sub(r'\s+', ' ', text)
            # 去除行首行尾空白
            text = text.strip()
        
        # 去除特殊字符
        if self.config.get("remove_special_chars", False):
            # 保留中文、英文、数字、常见标点
            text = re.sub(r'[^\w\s\u4e00-\u9fff.,;!?()（）。，；！？\[\]【】{}\-]', '', text)
        
        # 规范化 Unicode
        if self.config.get("normalize_unicode", True):
            import unicodedata
            text = unicodedata.normalize('NFC', text)
        
        # 规范化换行
        if self.config.get("trim_lines", True):
            lines = text.split('\n')
            lines = [line.strip() for line in lines]
            text = '\n'.join(lines)
        
        return text


class FormatConversionProcessor(BaseProcessor):
    """格式转换处理器
    
    支持 JSON、Markdown、HTML 格式之间的转换。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化格式转换处理器
        
        Args:
            config: 格式转换配置
        """
        self.config = config
        self.supported_formats = config.get("supported_formats", ["json", "markdown", "html"])
    
    def process(self, data: Any, **kwargs) -> Any:
        """转换数据格式
        
        Args:
            data: 待转换的数据
            **kwargs: 额外参数，包括：
                - target_format: 目标格式（json/markdown/html）
                
        Returns:
            转换后的数据
        """
        target_format = kwargs.get("target_format", self.config.get("default_format", "json"))
        
        if target_format not in self.supported_formats:
            raise ValueError(f"不支持的目标格式: {target_format}，支持的格式: {self.supported_formats}")
        
        if target_format == "json":
            return self._to_json(data)
        elif target_format == "markdown":
            return self._to_markdown(data)
        elif target_format == "html":
            return self._to_html(data)
        else:
            return data
    
    def _to_json(self, data: Any) -> str:
        """转换为 JSON 格式
        
        Args:
            data: 待转换的数据
            
        Returns:
            JSON 字符串
        """
        if isinstance(data, str):
            try:
                # 尝试解析已有的 JSON
                json.loads(data)
                return data
            except json.JSONDecodeError:
                # 不是 JSON，包装为 JSON
                return json.dumps({"content": data}, ensure_ascii=False, indent=2)
        else:
            return json.dumps(data, ensure_ascii=False, indent=2)
    
    def _to_markdown(self, data: Any) -> str:
        """转换为 Markdown 格式
        
        Args:
            data: 待转换的数据
            
        Returns:
            Markdown 字符串
        """
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return self._dict_to_markdown(data)
        elif isinstance(data, list):
            return self._list_to_markdown(data)
        else:
            return str(data)
    
    def _dict_to_markdown(self, data: Dict[str, Any], level: int = 0) -> str:
        """将字典转换为 Markdown
        
        Args:
            data: 字典数据
            level: 嵌套层级
            
        Returns:
            Markdown 字符串
        """
        lines = []
        for key, value in data.items():
            prefix = "  " * level
            if isinstance(value, dict):
                lines.append(f"{prefix}**{key}:**")
                lines.append(self._dict_to_markdown(value, level + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}**{key}:**")
                lines.append(self._list_to_markdown(value, level + 1))
            else:
                lines.append(f"{prefix}**{key}:** {value}")
        return "\n".join(lines)
    
    def _list_to_markdown(self, data: List[Any], level: int = 0) -> str:
        """将列表转换为 Markdown
        
        Args:
            data: 列表数据
            level: 嵌套层级
            
        Returns:
            Markdown 字符串
        """
        lines = []
        prefix = "  " * level
        for item in data:
            if isinstance(item, dict):
                lines.append(f"{prefix}- ")
                lines.append(self._dict_to_markdown(item, level + 1))
            elif isinstance(item, list):
                lines.append(self._list_to_markdown(item, level + 1))
            else:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines)
    
    def _to_html(self, data: Any) -> str:
        """转换为 HTML 格式
        
        Args:
            data: 待转换的数据
            
        Returns:
            HTML 字符串
        """
        if isinstance(data, str):
            # 转义 HTML 特殊字符
            return html.escape(data)
        elif isinstance(data, dict):
            return self._dict_to_html(data)
        elif isinstance(data, list):
            return self._list_to_html(data)
        else:
            return html.escape(str(data))
    
    def _dict_to_html(self, data: Dict[str, Any]) -> str:
        """将字典转换为 HTML
        
        Args:
            data: 字典数据
            
        Returns:
            HTML 字符串
        """
        lines = ["<dl>"]
        for key, value in data.items():
            lines.append(f"  <dt>{html.escape(str(key))}</dt>")
            if isinstance(value, (dict, list)):
                lines.append(f"  <dd>{self._to_html(value)}</dd>")
            else:
                lines.append(f"  <dd>{html.escape(str(value))}</dd>")
        lines.append("</dl>")
        return "\n".join(lines)
    
    def _list_to_html(self, data: List[Any]) -> str:
        """将列表转换为 HTML
        
        Args:
            data: 列表数据
            
        Returns:
            HTML 字符串
        """
        lines = ["<ul>"]
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"  <li>{self._to_html(item)}</li>")
            else:
                lines.append(f"  <li>{html.escape(str(item))}</li>")
        lines.append("</ul>")
        return "\n".join(lines)


class FieldFilterProcessor(BaseProcessor):
    """字段过滤处理器
    
    支持字段过滤和重命名。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化字段过滤处理器
        
        Args:
            config: 字段过滤配置
        """
        self.config = config
        self.include_fields = config.get("include_fields", [])
        self.exclude_fields = config.get("exclude_fields", [])
        self.rename_fields = config.get("rename_fields", {})
    
    def process(self, data: Any, **kwargs) -> Any:
        """过滤和重命名字段
        
        Args:
            data: 待处理的数据
            **kwargs: 额外参数
            
        Returns:
            处理后的数据
        """
        if isinstance(data, dict):
            return self._process_dict(data)
        elif isinstance(data, list):
            return [self.process(item, **kwargs) for item in data]
        else:
            return data
    
    def _process_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理字典数据
        
        Args:
            data: 字典数据
            
        Returns:
            处理后的字典
        """
        result = {}
        
        for key, value in data.items():
            # 检查是否应该排除
            if self.exclude_fields and key in self.exclude_fields:
                continue
            
            # 检查是否应该包含（如果指定了包含列表）
            if self.include_fields and key not in self.include_fields:
                continue
            
            # 重命名字段
            new_key = self.rename_fields.get(key, key)
            
            # 递归处理嵌套字典
            if isinstance(value, dict):
                result[new_key] = self._process_dict(value)
            elif isinstance(value, list):
                result[new_key] = [
                    self._process_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[new_key] = value
        
        return result


class CustomRuleEngine(BaseProcessor):
    """自定义规则引擎
    
    支持通过配置定义自定义后处理规则。
    """
    
    def __init__(self, rules: List[Dict[str, Any]]):
        """初始化规则引擎
        
        Args:
            rules: 规则列表，每个规则是一个字典，包含：
                - name: 规则名称
                - type: 规则类型（regex_replace/field_mapping/custom）
                - config: 规则配置
        """
        self.rules = rules
        self._custom_handlers: Dict[str, Callable] = {}
    
    def register_handler(self, name: str, handler: Callable) -> None:
        """注册自定义处理函数
        
        Args:
            name: 处理函数名称
            handler: 处理函数
        """
        self._custom_handlers[name] = handler
    
    def process(self, data: Any, **kwargs) -> Any:
        """应用所有规则
        
        Args:
            data: 待处理的数据
            **kwargs: 额外参数
            
        Returns:
            处理后的数据
        """
        result = deepcopy(data)
        
        for rule in self.rules:
            rule_type = rule.get("type")
            rule_config = rule.get("config", {})
            
            if rule_type == "regex_replace":
                result = self._apply_regex_replace(result, rule_config)
            elif rule_type == "field_mapping":
                result = self._apply_field_mapping(result, rule_config)
            elif rule_type == "custom":
                result = self._apply_custom_rule(result, rule_config)
            else:
                raise ValueError(f"未知的规则类型: {rule_type}")
        
        return result
    
    def _apply_regex_replace(self, data: Any, config: Dict[str, Any]) -> Any:
        """应用正则表达式替换规则
        
        Args:
            data: 数据
            config: 规则配置，包含 pattern 和 replacement
            
        Returns:
            处理后的数据
        """
        pattern = config.get("pattern")
        replacement = config.get("replacement", "")
        field = config.get("field")
        
        if not pattern:
            return data
        
        if isinstance(data, str):
            return re.sub(pattern, replacement, data)
        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if field and key != field:
                    result[key] = value
                else:
                    result[key] = self._apply_regex_replace(value, config)
            return result
        elif isinstance(data, list):
            return [self._apply_regex_replace(item, config) for item in data]
        else:
            return data
    
    def _apply_field_mapping(self, data: Any, config: Dict[str, Any]) -> Any:
        """应用字段映射规则
        
        Args:
            data: 数据
            config: 规则配置，包含 mappings 字典
            
        Returns:
            处理后的数据
        """
        mappings = config.get("mappings", {})
        
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                new_key = mappings.get(key, key)
                result[new_key] = self._apply_field_mapping(value, config)
            return result
        elif isinstance(data, list):
            return [self._apply_field_mapping(item, config) for item in data]
        else:
            return data
    
    def _apply_custom_rule(self, data: Any, config: Dict[str, Any]) -> Any:
        """应用自定义规则
        
        Args:
            data: 数据
            config: 规则配置，包含 handler_name
            
        Returns:
            处理后的数据
        """
        handler_name = config.get("handler_name")
        
        if not handler_name:
            return data
        
        handler = self._custom_handlers.get(handler_name)
        if not handler:
            raise ValueError(f"未找到自定义处理函数: {handler_name}")
        
        return handler(data, **config.get("params", {}))


class ResultProcessor:
    """结果处理器
    
    整合所有后处理逻辑，提供统一的处理接口。
    支持插件式扩展。
    """
    
    def __init__(self, config: PostprocessConfig):
        """初始化结果处理器
        
        Args:
            config: 后处理配置
        """
        self.config = config
        self._processors: List[BaseProcessor] = []
        self._custom_processors: Dict[str, BaseProcessor] = {}
        
        # 初始化内置处理器
        self._init_builtin_processors()
    
    def _init_builtin_processors(self) -> None:
        """初始化内置处理器"""
        # 文本清理处理器
        if self.config.enable_text_cleaning:
            self._processors.append(
                TextCleaningProcessor(self.config.text_cleaning)
            )
        
        # 字段过滤处理器
        if self.config.enable_field_filter:
            self._processors.append(
                FieldFilterProcessor(self.config.field_filter)
            )
        
        # 自定义规则引擎
        if self.config.custom_rules:
            self._processors.append(
                CustomRuleEngine(self.config.custom_rules)
            )
    
    def register_processor(self, name: str, processor: BaseProcessor) -> None:
        """注册自定义处理器
        
        Args:
            name: 处理器名称
            processor: 处理器实例
        """
        self._custom_processors[name] = processor
        self._processors.append(processor)
    
    def unregister_processor(self, name: str) -> None:
        """注销自定义处理器
        
        Args:
            name: 处理器名称
        """
        if name in self._custom_processors:
            processor = self._custom_processors.pop(name)
            self._processors.remove(processor)
    
    def process(
        self,
        data: Any,
        target_format: Optional[str] = None,
        **kwargs
    ) -> Any:
        """处理数据
        
        按照注册的处理器顺序依次处理数据。
        
        Args:
            data: 待处理的数据
            target_format: 目标格式（json/markdown/html）
            **kwargs: 额外的处理参数
            
        Returns:
            处理后的数据
        """
        result = deepcopy(data)
        
        # 应用所有处理器
        for processor in self._processors:
            result = processor.process(result, **kwargs)
        
        # 格式转换（如果指定了目标格式）
        if target_format and self.config.enable_format_conversion:
            format_processor = FormatConversionProcessor(self.config.format_conversion)
            result = format_processor.process(result, target_format=target_format)
        
        return result
    
    def process_batch(
        self,
        data_list: List[Any],
        target_format: Optional[str] = None,
        **kwargs
    ) -> List[Any]:
        """批量处理数据
        
        Args:
            data_list: 待处理的数据列表
            target_format: 目标格式（json/markdown/html）
            **kwargs: 额外的处理参数
            
        Returns:
            处理后的数据列表
        """
        return [
            self.process(data, target_format=target_format, **kwargs)
            for data in data_list
        ]
