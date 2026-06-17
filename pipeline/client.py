"""各微服务的 HTTP 客户端封装"""

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class ServiceClientError(Exception):
    """服务调用错误"""
    pass


class PreprocessClient:
    """预处理服务客户端"""

    def __init__(self, config: PipelineConfig):
        self.base_url = config.preprocess.base_url
        self.timeout = httpx.Timeout(60.0, connect=10.0)

    async def preprocess(
        self,
        file_path: str,
        return_intermediate: bool = False,
    ) -> Dict[str, Any]:
        """调用预处理流水线接口

        Args:
            file_path: 图像文件路径
            return_intermediate: 是否返回中间结果

        Returns:
            包含 final_image (base64) 等字段的字典

        Raises:
            ServiceClientError: 调用失败
        """
        path = Path(file_path)
        if not path.exists():
            raise ServiceClientError(f"文件不存在: {file_path}")

        url = f"{self.base_url}/preprocess"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                with open(path, "rb") as f:
                    files = {"file": (path.name, f, "image/jpeg")}
                    data = {"return_intermediate": str(return_intermediate).lower()}
                    response = await client.post(url, files=files, data=data)
                    response.raise_for_status()
                    return response.json()
        except httpx.HTTPError as e:
            raise ServiceClientError(f"预处理服务调用失败: {e}") from e

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False


class GLMOCRClient:
    """GLM-OCR 异步服务客户端

    支持两种等待模式：
    - WebSocket（默认）：连接 /ws/{doc_id}，服务端主动推送进度和结果，零无效查询
    - HTTP 轮询（降级）：当 WebSocket 不可用时自动降级为轮询
    """

    def __init__(self, config: PipelineConfig):
        self.base_url = config.glmocr.base_url
        self.timeout = httpx.Timeout(60.0, connect=10.0)
        self.poll_interval = config.poll_interval
        self.poll_timeout = config.poll_timeout

    async def submit(
        self,
        image_b64: str,
        filename: str = "image.jpg",
    ) -> str:
        """提交异步 OCR 任务

        Args:
            image_b64: base64 编码的图像
            filename: 文件名

        Returns:
            doc_id

        Raises:
            ServiceClientError: 提交失败
        """
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            raise ServiceClientError(f"base64 解码失败: {e}") from e

        url = f"{self.base_url}/parse/async"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                files = {"file": (filename, io.BytesIO(image_bytes), "image/jpeg")}
                response = await client.post(url, files=files)
                response.raise_for_status()
                data = response.json()
                return data["doc_id"]
        except httpx.HTTPError as e:
            raise ServiceClientError(f"GLM-OCR 提交失败: {e}") from e
        except KeyError as e:
            raise ServiceClientError(f"GLM-OCR 响应缺少 doc_id: {e}") from e

    async def get_status(self, doc_id: str) -> Dict[str, Any]:
        """查询任务状态

        Args:
            doc_id: 文档 ID

        Returns:
            包含 completed/total/status 的字典
        """
        url = f"{self.base_url}/parse/status/{doc_id}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise ServiceClientError(f"查询状态失败: {e}") from e

    async def get_result(self, doc_id: str) -> Dict[str, Any]:
        """获取 OCR 结果

        Args:
            doc_id: 文档 ID

        Returns:
            OCR 结果字典
        """
        url = f"{self.base_url}/parse/result/{doc_id}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                if response.status_code == 202:
                    raise ServiceClientError("文档处理尚未完成")
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise ServiceClientError(f"获取结果失败: {e}") from e

    async def wait_for_completion(
        self,
        doc_id: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """等待任务完成（优先使用 WebSocket，降级为轮询）

        Args:
            doc_id: 文档 ID
            timeout: 总超时（秒）

        Returns:
            最终的 OCR 结果

        Raises:
            ServiceClientError: 超时或失败
        """
        timeout = timeout or self.poll_timeout
        try:
            return await self._wait_via_websocket(doc_id, timeout)
        except Exception as e:
            logger.warning(
                f"WebSocket 等待失败，降级为轮询: {e}"
            )
            return await self._wait_via_polling(doc_id, timeout)

    async def _wait_via_websocket(
        self,
        doc_id: str,
        timeout: float,
    ) -> Dict[str, Any]:
        """通过 WebSocket 等待任务完成

        连接 /ws/{doc_id}，服务端主动推送进度更新和最终结果。
        零无效查询，比轮询高效得多。

        Args:
            doc_id: 文档 ID
            timeout: 总超时（秒）

        Returns:
            最终的 OCR 结果
        """
        import websockets

        # http://host:port → ws://host:port
        ws_url = f"ws://{self.base_url.split('://', 1)[1]}/ws/{doc_id}"

        try:
            async with websockets.connect(
                ws_url,
                close_timeout=5,
                open_timeout=10,
            ) as ws:
                # 设置总超时
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        raise ServiceClientError(
                            f"WebSocket 等待超时 (>{timeout}s): {doc_id}"
                        )

                    import json
                    msg = json.loads(raw)
                    msg_type = msg.get("type")

                    if msg_type == "progress":
                        logger.info(
                            f"[doc_id={doc_id}] 进度: "
                            f"{msg['completed']}/{msg['total']}"
                        )
                    elif msg_type == "complete":
                        logger.info(f"[doc_id={doc_id}] 处理完成")
                        return msg["result"]
                    elif msg_type == "error":
                        raise ServiceClientError(
                            f"服务端错误: {msg.get('message', 'unknown')}"
                        )

        except ServiceClientError:
            raise
        except Exception as e:
            raise ServiceClientError(f"WebSocket 连接失败: {e}") from e

    async def _wait_via_polling(
        self,
        doc_id: str,
        timeout: float,
    ) -> Dict[str, Any]:
        """轮询等待任务完成（指数退避）

        Args:
            doc_id: 文档 ID
            timeout: 总超时（秒）

        Returns:
            最终的 OCR 结果
        """
        elapsed = 0.0
        interval = self.poll_interval
        max_interval = 5.0

        while elapsed < timeout:
            status = await self.get_status(doc_id)
            logger.info(
                f"[doc_id={doc_id}] 进度: {status['completed']}/{status['total']}, "
                f"状态: {status['status']}"
            )
            if status["status"] == "completed":
                return await self.get_result(doc_id)
            if status["status"] == "not_found":
                raise ServiceClientError(f"doc_id 不存在: {doc_id}")

            await asyncio.sleep(interval)
            elapsed += interval
            # 指数退避：逐步拉长轮询间隔
            interval = min(interval * 1.5, max_interval)

        raise ServiceClientError(f"任务超时 (>{timeout}s): {doc_id}")

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False


class PostprocessClient:
    """后处理服务客户端"""

    def __init__(self, config: PipelineConfig):
        self.base_url = config.postprocess.base_url
        self.timeout = httpx.Timeout(60.0, connect=10.0)

    async def postprocess(
        self,
        ocr_result: Dict[str, Any],
        target_format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """调用后处理接口

        Args:
            ocr_result: OCR 识别结果
            target_format: 目标格式
            options: 额外选项

        Returns:
            后处理结果
        """
        url = f"{self.base_url}/postprocess"
        payload = {
            "data": ocr_result,
            "target_format": target_format,
            "options": options or {},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise ServiceClientError(f"后处理服务调用失败: {e}") from e

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False
