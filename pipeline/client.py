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

    优先使用 WebSocket 监听完成事件，避免无效轮询。
    当 WebSocket 不可用时自动降级为 HTTP 轮询（指数退避）。
    """

    def __init__(self, config: PipelineConfig):
        self.base_url = config.glmocr.base_url
        self.timeout = httpx.Timeout(60.0, connect=10.0)

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
        """查询任务状态（HTTP 轮询）"""
        url = f"{self.base_url}/parse/status/{doc_id}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            raise ServiceClientError(f"查询状态失败: {e}") from e

    async def get_result(self, doc_id: str) -> Dict[str, Any]:
        """获取 OCR 结果"""
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
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """等待任务完成（优先 WebSocket，降级 HTTP 轮询）

        Args:
            doc_id: 文档 ID
            poll_interval: HTTP 轮询基础间隔（秒），仅降级时使用
            timeout: 总超时（秒）

        Returns:
            最终的 OCR 结果

        Raises:
            ServiceClientError: 超时或失败
        """
        try:
            return await self._wait_via_websocket(doc_id, timeout)
        except Exception as e:
            logger.warning(
                f"WebSocket 等待失败，降级为 HTTP 轮询: {e}"
            )
            return await self._wait_via_polling(doc_id, poll_interval, timeout)

    async def _wait_via_websocket(
        self,
        doc_id: str,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """通过 WebSocket 等待任务完成

        服务端 /ws/{doc_id} 会推送 progress 事件，
        当 completed >= total 时表示完成。
        """
        try:
            import websockets
        except ImportError:
            raise ServiceClientError("websockets 库未安装，请 pip install websockets")

        # http://host:port → ws://host:port
        ws_url = f"ws://{self.base_url.replace('http://', '').replace('https://', '')}/ws/{doc_id}"

        try:
            async with websockets.connect(
                ws_url,
                close_timeout=5,
                max_queue=256,
            ) as ws:
                elapsed = 0.0
                while elapsed < timeout:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=min(5.0, timeout - elapsed))
                    except asyncio.TimeoutError:
                        elapsed += 5.0
                        continue

                    import json
                    data = json.loads(msg)

                    if data.get("type") == "progress":
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)
                        logger.info(
                            f"[doc_id={doc_id}] WebSocket 进度: {completed}/{total}"
                        )
                        if total > 0 and completed >= total:
                            return await self.get_result(doc_id)

                    elif data.get("type") == "complete":
                        logger.info(f"[doc_id={doc_id}] WebSocket 收到完成信号")
                        return await self.get_result(doc_id)

                    elif data.get("type") == "error":
                        raise ServiceClientError(
                            f"服务端报错: {data.get('message', 'unknown')}"
                        )

                raise ServiceClientError(f"WebSocket 等待超时 (>{timeout}s): {doc_id}")

        except ServiceClientError:
            raise
        except Exception as e:
            raise ServiceClientError(f"WebSocket 连接失败: {e}") from e

    async def _wait_via_polling(
        self,
        doc_id: str,
        base_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """HTTP 轮询等待任务完成（指数退避）

        前几次快速轮询，之后逐步拉长间隔：
          第 1-5 次: base_interval (1s)
          第 6-10 次: 2s
          第 11+ 次: 3s
        """
        elapsed = 0.0
        attempt = 0

        while elapsed < timeout:
            status = await self.get_status(doc_id)
            logger.info(
                f"[doc_id={doc_id}] 轮询进度: {status['completed']}/{status['total']}, "
                f"状态: {status['status']}"
            )

            if status["status"] == "completed":
                return await self.get_result(doc_id)
            if status["status"] == "not_found":
                raise ServiceClientError(f"doc_id 不存在: {doc_id}")

            # 指数退避：前5次1s，6-10次2s，之后3s
            attempt += 1
            if attempt <= 5:
                interval = base_interval
            elif attempt <= 10:
                interval = base_interval * 2
            else:
                interval = base_interval * 3

            await asyncio.sleep(interval)
            elapsed += interval

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
