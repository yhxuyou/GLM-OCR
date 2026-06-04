from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

import httpx


class FileFetchError(Exception):
    """文件拉取失败异常，包含 file_id 和 seq 信息"""

    def __init__(self, file_id: str, seq: int, message: str = "") -> None:
        self.file_id = file_id
        self.seq = seq
        super().__init__(f"文件拉取失败 [file_id={file_id}, seq={seq}]: {message}")


class UnsupportedFileType(Exception):
    """不支持的文件类型异常"""

    def __init__(self, file_type: str) -> None:
        self.file_type = file_type
        super().__init__(f"不支持的文件类型: {file_type}")


@runtime_checkable
class FileFetcher(Protocol):
    """文件拉取器协议，定义拉取文件的接口"""

    def fetch(self, file_id: str, seq: int) -> bytes:
        """根据 file_id 和 seq 拉取文件内容，返回字节数据"""
        ...


class FileIdFetcher:
    """基于 file_id 的文件拉取器，通过拼接 URL 使用 httpx 同步 GET 请求下载图片字节"""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        retries: int = 2,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.headers = headers or {}

    def fetch(self, file_id: str, seq: int) -> bytes:
        """根据 file_id 和 seq 拼接 URL 并下载文件内容

        拼接规则: {base_url}/{file_id}/{seq}
        超时、404、5xx 均抛出 FileFetchError，支持重试
        """
        url = f"{self.base_url}/{file_id}/{seq}"
        last_error: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
                    response = client.get(url)

                if response.status_code >= 200 and response.status_code < 300:
                    return response.content

                # 404 或 5xx 视为失败
                last_error = Exception(
                    f"HTTP 状态码 {response.status_code}"
                )
            except httpx.TimeoutException as exc:
                last_error = exc
            except httpx.HTTPError as exc:
                last_error = exc

            # 未成功且还有重试机会时，等待后重试
            if attempt < self.retries:
                time.sleep(0.5)

        raise FileFetchError(
            file_id, seq, str(last_error) if last_error else "未知错误"
        )


class FileFetcherRegistry:
    """文件拉取器注册表，根据文件类型分发到对应的拉取器"""

    def __init__(
        self,
        supported_types: List[str],
        fetcher_factory: Dict[str, Callable[..., FileFetcher]],
    ) -> None:
        self.supported_types = supported_types
        self._registry: Dict[str, FileFetcher] = {}
        # 通过工厂函数创建并注册拉取器
        for file_type, factory in fetcher_factory.items():
            self._registry[file_type] = factory()

    def register(self, file_type: str, fetcher: FileFetcher) -> None:
        """注册指定文件类型的拉取器"""
        self._registry[file_type] = fetcher

    def get_fetcher(self, file_type: str) -> FileFetcher:
        """获取指定文件类型的拉取器，未注册则抛出 UnsupportedFileType"""
        if file_type not in self._registry:
            raise UnsupportedFileType(file_type)
        return self._registry[file_type]

    def fetch(self, file_type: str, file_id: str, seq: int) -> bytes:
        """便捷方法：根据文件类型获取对应拉取器并拉取文件"""
        fetcher = self.get_fetcher(file_type)
        return fetcher.fetch(file_id, seq)
