"""Pipeline 核心编排服务

整合预处理、GLM-OCR 异步处理、后处理三个微服务，提供端到端文档解析能力。
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional

from .client import (
    GLMOCRClient,
    PostprocessClient,
    PreprocessClient,
    ServiceClientError,
)
from .config import PipelineConfig, config

logger = logging.getLogger(__name__)


class PipelineService:
    """Pipeline 编排服务

    串联执行：
        1. 预处理（文档检测 → 方向矫正 → 扭曲矫正）
        2. GLM-OCR 异步识别
        3. 后处理
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.config = cfg or config
        self.preprocess_client = PreprocessClient(self.config)
        self.glmocr_client = GLMOCRClient(self.config)
        self.postprocess_client = PostprocessClient(self.config)

    async def run_preprocess(self, file_path: str) -> str:
        """Step 1: 预处理

        Args:
            file_path: 输入文件路径

        Returns:
            base64 编码的预处理后图像
        """
        logger.info(f"[Pipeline] Step 1: 预处理 - {file_path}")
        result = await self.preprocess_client.preprocess(file_path)
        image_b64 = result.get("final_image")
        if not image_b64:
            raise ServiceClientError("预处理响应缺少 final_image 字段")
        return image_b64

    async def run_glmocr(
        self,
        image_b64: str,
        filename: str = "preprocessed.jpg",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Step 2: GLM-OCR 异步处理

        Args:
            image_b64: 预处理后的 base64 图像
            filename: 文件名
            timeout: 超时时间

        Returns:
            OCR 识别结果
        """
        logger.info("[Pipeline] Step 2: GLM-OCR 异步处理")

        # 2.1 提交
        doc_id = await self.glmocr_client.submit(image_b64, filename=filename)
        logger.info(f"[Pipeline] 提交成功, doc_id={doc_id}")

        # 2.2 轮询等待完成
        timeout = timeout or self.config.poll_timeout
        result = await self.glmocr_client.wait_for_completion(
            doc_id,
            poll_interval=self.config.poll_interval,
            timeout=timeout,
        )
        logger.info(f"[Pipeline] GLM-OCR 完成, doc_id={doc_id}")
        return result

    async def run_postprocess(
        self,
        ocr_result: Dict[str, Any],
        target_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Step 3: 后处理

        Args:
            ocr_result: OCR 识别结果
            target_format: 目标格式

        Returns:
            后处理结果
        """
        logger.info("[Pipeline] Step 3: 后处理")
        result = await self.postprocess_client.postprocess(
            ocr_result, target_format=target_format
        )
        return result

    async def process(
        self,
        file_path: str,
        target_format: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """端到端处理流程

        Args:
            file_path: 输入文件路径
            target_format: 后处理目标格式
            timeout: OCR 阶段超时时间

        Returns:
            包含各阶段结果的字典
        """
        task_id = str(uuid.uuid4())
        start_time = time.time()
        logger.info(f"[Pipeline task_id={task_id}] 开始处理: {file_path}")

        try:
            # Step 1: 预处理
            preprocess_b64 = await self.run_preprocess(file_path)

            # Step 2: GLM-OCR
            ocr_result = await self.run_glmocr(
                preprocess_b64, filename=file_path.split("/")[-1], timeout=timeout
            )

            # Step 3: 后处理
            postprocess_result = await self.run_postprocess(
                ocr_result, target_format=target_format
            )

            elapsed = time.time() - start_time
            logger.info(
                f"[Pipeline task_id={task_id}] 处理完成, 耗时 {elapsed:.2f}s"
            )

            return {
                "task_id": task_id,
                "status": "success",
                "elapsed": elapsed,
                "preprocess": {"status": "success"},
                "ocr_result": ocr_result,
                "postprocess": postprocess_result,
            }

        except ServiceClientError as e:
            elapsed = time.time() - start_time
            logger.error(f"[Pipeline task_id={task_id}] 处理失败: {e}")
            return {
                "task_id": task_id,
                "status": "failed",
                "elapsed": elapsed,
                "error": str(e),
            }

    async def process_base64(
        self,
        image_b64: str,
        filename: str = "image.jpg",
        target_format: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """跳过预处理的端到端流程（输入已经是 base64 图像）

        Args:
            image_b64: base64 编码的图像
            filename: 文件名
            target_format: 后处理目标格式
            timeout: OCR 阶段超时时间

        Returns:
            包含各阶段结果的字典
        """
        task_id = str(uuid.uuid4())
        start_time = time.time()
        logger.info(f"[Pipeline task_id={task_id}] 开始处理 (base64): {filename}")

        try:
            # Step 2: GLM-OCR
            ocr_result = await self.run_glmocr(
                image_b64, filename=filename, timeout=timeout
            )

            # Step 3: 后处理
            postprocess_result = await self.run_postprocess(
                ocr_result, target_format=target_format
            )

            elapsed = time.time() - start_time
            return {
                "task_id": task_id,
                "status": "success",
                "elapsed": elapsed,
                "ocr_result": ocr_result,
                "postprocess": postprocess_result,
            }

        except ServiceClientError as e:
            elapsed = time.time() - start_time
            logger.error(f"[Pipeline task_id={task_id}] 处理失败: {e}")
            return {
                "task_id": task_id,
                "status": "failed",
                "elapsed": elapsed,
                "error": str(e),
            }

    async def health_check(self) -> Dict[str, bool]:
        """检查所有依赖服务的健康状态"""
        preprocess_ok, glmocr_ok, postprocess_ok = await asyncio.gather(
            self.preprocess_client.health_check(),
            self.glmocr_client.health_check(),
            self.postprocess_client.health_check(),
            return_exceptions=True,
        )
        return {
            "preprocess": preprocess_ok is True,
            "glmocr": glmocr_ok is True,
            "postprocess": postprocess_ok is True,
        }
