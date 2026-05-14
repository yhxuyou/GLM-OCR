"""
OCR处理器模块
集成GLM-OCR核心库
"""
import os
import sys
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import threading

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from glmocr import GlmOcr
    from glmocr.config import load_config
    GLMOCR_AVAILABLE = True
except ImportError:
    GLMOCR_AVAILABLE = False

from .config import get_settings
from .logger import logger


class OCRProcessor:
    """OCR处理器单例"""
    
    _instance: Optional['OCRProcessor'] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            self.settings = get_settings()
            self._glm_ocr: Optional[GlmOcr] = None
            self._lock = threading.Lock()
            self._initialized = True
    
    def initialize(self):
        """初始化GLM-OCR"""
        if not GLMOCR_AVAILABLE:
            raise ImportError("GLM-OCR library not available")
        
        with self._lock:
            if self._glm_ocr is not None:
                return
            
            logger.info("Initializing GLM-OCR...")
            
            try:
                # 使用配置初始化GlmOcr
                self._glm_ocr = GlmOcr(
                    config_path=self.settings.GLMOCR_CONFIG_PATH,
                    api_key=self.settings.GLMOCR_API_KEY,
                    api_url=self.settings.GLMOCR_API_URL,
                    model=self.settings.GLMOCR_MODEL,
                    mode=self.settings.GLMOCR_MODE,
                )
                
                logger.info("GLM-OCR initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize GLM-OCR: {e}", exc_info=True)
                raise
    
    def shutdown(self):
        """关闭OCR处理器"""
        with self._lock:
            if self._glm_ocr is not None:
                try:
                    self._glm_ocr.close()
                    logger.info("GLM-OCR shutdown successfully")
                except Exception as e:
                    logger.error(f"Error during GLM-OCR shutdown: {e}", exc_info=True)
                self._glm_ocr = None
    
    def process_file(
        self,
        file_path: Union[str, Path],
        save_layout_visualization: bool = False,
        stream: bool = False
    ) -> Dict[str, Any]:
        """处理单个文件"""
        if self._glm_ocr is None:
            raise RuntimeError("OCR processor not initialized")
        
        try:
            logger.info(f"Processing file: {file_path}")
            
            # 调用GLM-OCR处理
            result = self._glm_ocr.parse(
                str(file_path),
                stream=stream,
                save_layout_visualization=save_layout_visualization
            )
            
            # 处理结果
            if stream:
                # 流式处理，收集所有结果
                results = list(result)
                if len(results) == 1:
                    return self._format_result(results[0])
                return self._format_batch_result(results)
            else:
                return self._format_result(result)
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            raise
    
    def process_bytes(
        self,
        file_bytes: bytes,
        file_extension: str,
        save_layout_visualization: bool = False
    ) -> Dict[str, Any]:
        """处理字节流"""
        if self._glm_ocr is None:
            raise RuntimeError("OCR processor not initialized")
        
        try:
            # 保存为临时文件
            temp_dir = self.settings.UPLOAD_DIR
            temp_path = temp_dir / f"{uuid.uuid4()}.{file_extension}"
            
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            
            try:
                # 处理文件
                return self.process_file(
                    temp_path,
                    save_layout_visualization=save_layout_visualization
                )
            finally:
                # 清理临时文件
                if temp_path.exists():
                    temp_path.unlink()
                    
        except Exception as e:
            logger.error(f"Error processing bytes: {e}", exc_info=True)
            raise
    
    def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        save_layout_visualization: bool = False
    ) -> List[Dict[str, Any]]:
        """批量处理文件"""
        if self._glm_ocr is None:
            raise RuntimeError("OCR processor not initialized")
        
        try:
            logger.info(f"Processing batch of {len(file_paths)} files")
            
            results = self._glm_ocr.parse(
                [str(p) for p in file_paths],
                save_layout_visualization=save_layout_visualization
            )
            
            return [self._format_result(r) for r in results]
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            raise
    
    def _format_result(self, result) -> Dict[str, Any]:
        """格式化单个结果"""
        return {
            "json_result": result.json_result if hasattr(result, 'json_result') else None,
            "markdown_result": result.markdown_result if hasattr(result, 'markdown_result') else None,
            "original_images": result.original_images if hasattr(result, 'original_images') else [],
            "image_files": result.image_files if hasattr(result, 'image_files') else None,
            "raw_json_result": result.raw_json_result if hasattr(result, 'raw_json_result') else None,
            "layout_vis_images": (
                {k: str(v) for k, v in result.layout_vis_images.items()}
                if hasattr(result, 'layout_vis_images') and result.layout_vis_images
                else None
            ),
            "error": result._error if hasattr(result, '_error') else None
        }
    
    def _format_batch_result(self, results: List) -> Dict[str, Any]:
        """格式化批量结果"""
        return {
            "results": [self._format_result(r) for r in results],
            "count": len(results)
        }
    
    def get_queue_stats(self) -> Optional[Dict[str, int]]:
        """获取GLM-OCR内部队列统计"""
        if self._glm_ocr:
            return self._glm_ocr.get_queue_stats()
        return None
    
    def is_healthy(self) -> bool:
        """检查健康状态"""
        return self._glm_ocr is not None


def get_ocr_processor() -> OCRProcessor:
    """获取OCR处理器单例"""
    return OCRProcessor()
