from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from glmocr.file_fetcher import (
    FileFetchError,
    FileFetcherRegistry,
    FileIdFetcher,
    UnsupportedFileType,
)


# ---------------------------------------------------------------------------
# FileIdFetcher 测试
# ---------------------------------------------------------------------------


class TestFileIdFetcher:
    """FileIdFetcher 单测集合"""

    def test_url_format(self):
        """验证 fetch 拼接的 URL 是 {base_url}/{file_id}/{seq} 格式"""
        fetcher = FileIdFetcher(base_url="https://example.com/api", retries=0)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-content"

        with patch("glmocr.file_fetcher.httpx.Client") as MockClient:
            # 配置 mock client 的上下文管理器行为
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__enter__.return_value = mock_client

            fetcher.fetch(file_id="abc123", seq=0)

            # 验证请求的 URL 格式
            mock_client.get.assert_called_once_with("https://example.com/api/abc123/0")

    def test_url_format_trailing_slash(self):
        """验证 base_url 末尾斜杠会被去除后再拼接"""
        fetcher = FileIdFetcher(base_url="https://example.com/api/", retries=0)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"data"

        with patch("glmocr.file_fetcher.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__enter__.return_value = mock_client

            fetcher.fetch(file_id="xyz", seq=3)

            mock_client.get.assert_called_once_with("https://example.com/api/xyz/3")

    def test_successful_download(self):
        """mock 返回 200 + content，验证返回 bytes"""
        fetcher = FileIdFetcher(base_url="https://example.com", retries=0)

        expected_bytes = b"\x89PNG\r\n\x1a\nimage-data"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = expected_bytes

        with patch("glmocr.file_fetcher.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__enter__.return_value = mock_client

            result = fetcher.fetch(file_id="img1", seq=1)

            assert result == expected_bytes

    def test_2xx_status_codes(self):
        """验证 2xx 状态码均视为成功"""
        fetcher = FileIdFetcher(base_url="https://example.com", retries=0)

        for status_code in (200, 201, 204, 299):
            mock_response = MagicMock()
            mock_response.status_code = status_code
            mock_response.content = b"ok"

            with patch("glmocr.file_fetcher.httpx.Client") as MockClient:
                mock_client = MagicMock()
                mock_client.get.return_value = mock_response
                MockClient.return_value.__enter__.return_value = mock_client

                result = fetcher.fetch(file_id="f", seq=0)
                assert result == b"ok"

    @patch("glmocr.file_fetcher.time.sleep")
    def test_timeout_retry(self, mock_sleep):
        """mock httpx 抛 TimeoutException，验证重试 retries 次后抛 FileFetchError"""
        retries = 3
        fetcher = FileIdFetcher(base_url="https://example.com", retries=retries)

        with patch("glmocr.file_fetcher.httpx.Client") as MockClient:
            mock_client = MagicMock()
            # 每次请求都超时
            mock_client.get.side_effect = __import__("httpx").TimeoutException("timeout")
            MockClient.return_value.__enter__.return_value = mock_client

            with pytest.raises(FileFetchError) as exc_info:
                fetcher.fetch(file_id="timeout_file", seq=2)

            # 总共尝试 retries + 1 次
            assert mock_client.get.call_count == retries + 1
            # 验证异常信息包含 file_id 和 seq
            assert "timeout_file" in str(exc_info.value)
            assert "2" in str(exc_info.value)
            # 验证重试间有 sleep 调用（最后一次失败后不再 sleep）
            assert mock_sleep.call_count == retries

    @patch("glmocr.file_fetcher.time.sleep")
    def test_timeout_retry_default_retries(self, mock_sleep):
        """默认 retries=2 时，超时重试总共 3 次请求"""
        fetcher = FileIdFetcher(base_url="https://example.com", retries=2)

        with patch("glmocr.file_fetcher.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.side_effect = __import__("httpx").TimeoutException("timeout")
            MockClient.return_value.__enter__.return_value = mock_client

            with pytest.raises(FileFetchError):
                fetcher.fetch(file_id="f", seq=0)

            assert mock_client.get.call_count == 3  # 2 + 1

    @patch("glmocr.file_fetcher.time.sleep")
    def test_404_failure(self, mock_sleep):
        """mock 返回 404，验证抛 FileFetchError"""
        fetcher = FileIdFetcher(base_url="https://example.com", retries=1)

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.content = b"not found"

        with patch("glmocr.file_fetcher.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__enter__.return_value = mock_client

            with pytest.raises(FileFetchError) as exc_info:
                fetcher.fetch(file_id="missing", seq=5)

            # 404 也会重试 retries + 1 次
            assert mock_client.get.call_count == 2  # retries=1 → 2 次
            assert "missing" in str(exc_info.value)
            assert "404" in str(exc_info.value)

    @patch("glmocr.file_fetcher.time.sleep")
    def test_5xx_failure(self, mock_sleep):
        """mock 返回 500，验证抛 FileFetchError"""
        fetcher = FileIdFetcher(base_url="https://example.com", retries=0)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.content = b"internal error"

        with patch("glmocr.file_fetcher.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__enter__.return_value = mock_client

            with pytest.raises(FileFetchError) as exc_info:
                fetcher.fetch(file_id="err", seq=0)

            assert "500" in str(exc_info.value)

    @patch("glmocr.file_fetcher.time.sleep")
    def test_timeout_then_success(self, mock_sleep):
        """第一次超时，第二次成功，验证能正常返回"""
        fetcher = FileIdFetcher(base_url="https://example.com", retries=2)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"recovered"

        with patch("glmocr.file_fetcher.httpx.Client") as MockClient:
            mock_client = MagicMock()
            # 第一次超时，第二次成功
            mock_client.get.side_effect = [
                __import__("httpx").TimeoutException("timeout"),
                mock_response,
            ]
            MockClient.return_value.__enter__.return_value = mock_client

            result = fetcher.fetch(file_id="flaky", seq=0)

            assert result == b"recovered"
            assert mock_client.get.call_count == 2


# ---------------------------------------------------------------------------
# FileFetcherRegistry 测试
# ---------------------------------------------------------------------------


class TestFileFetcherRegistry:
    """FileFetcherRegistry 单测集合"""

    @staticmethod
    def _make_fetcher(return_value: bytes = b"data"):
        """创建一个 mock fetcher 工厂函数"""
        fetcher = MagicMock()
        fetcher.fetch.return_value = return_value
        return fetcher

    def test_get_fetcher_registered(self):
        """注册后能获取对应 fetcher"""
        mock_fetcher = self._make_fetcher()
        registry = FileFetcherRegistry(
            supported_types=["image"],
            fetcher_factory={"image": lambda: mock_fetcher},
        )

        result = registry.get_fetcher("image")
        assert result is mock_fetcher

    def test_get_fetcher_unregistered_raises(self):
        """未注册类型抛 UnsupportedFileType"""
        registry = FileFetcherRegistry(
            supported_types=["image"],
            fetcher_factory={"image": lambda: MagicMock()},
        )

        with pytest.raises(UnsupportedFileType) as exc_info:
            registry.get_fetcher("video")

        assert exc_info.value.file_type == "video"
        assert "video" in str(exc_info.value)

    def test_fetch_delegates_to_fetcher(self):
        """便捷方法 fetch 能正确调用对应 fetcher 的 fetch"""
        mock_fetcher = self._make_fetcher(return_value=b"image-bytes")
        registry = FileFetcherRegistry(
            supported_types=["image"],
            fetcher_factory={"image": lambda: mock_fetcher},
        )

        result = registry.fetch("image", file_id="pic1", seq=2)

        mock_fetcher.fetch.assert_called_once_with("pic1", 2)
        assert result == b"image-bytes"

    def test_fetch_unregistered_type_raises(self):
        """fetch 便捷方法对未注册类型也抛 UnsupportedFileType"""
        registry = FileFetcherRegistry(
            supported_types=[],
            fetcher_factory={},
        )

        with pytest.raises(UnsupportedFileType):
            registry.fetch("unknown", file_id="x", seq=0)

    def test_register_new_type(self):
        """动态注册新类型后能获取并使用"""
        registry = FileFetcherRegistry(
            supported_types=["image"],
            fetcher_factory={"image": lambda: self._make_fetcher()},
        )

        # 动态注册 pdf 类型
        pdf_fetcher = self._make_fetcher(return_value=b"pdf-bytes")
        registry.register("pdf", pdf_fetcher)

        # 验证能获取
        assert registry.get_fetcher("pdf") is pdf_fetcher

        # 验证能通过 fetch 使用
        result = registry.fetch("pdf", file_id="doc1", seq=0)
        pdf_fetcher.fetch.assert_called_once_with("doc1", 0)
        assert result == b"pdf-bytes"

    def test_register_overwrite(self):
        """动态注册覆盖已有类型"""
        old_fetcher = self._make_fetcher(return_value=b"old")
        registry = FileFetcherRegistry(
            supported_types=["image"],
            fetcher_factory={"image": lambda: old_fetcher},
        )

        new_fetcher = self._make_fetcher(return_value=b"new")
        registry.register("image", new_fetcher)

        assert registry.get_fetcher("image") is new_fetcher
        result = registry.fetch("image", file_id="f", seq=0)
        assert result == b"new"

    def test_multiple_types(self):
        """注册多个类型后能分别获取"""
        image_fetcher = self._make_fetcher(return_value=b"img")
        video_fetcher = self._make_fetcher(return_value=b"vid")

        registry = FileFetcherRegistry(
            supported_types=["image", "video"],
            fetcher_factory={
                "image": lambda: image_fetcher,
                "video": lambda: video_fetcher,
            },
        )

        assert registry.get_fetcher("image") is image_fetcher
        assert registry.get_fetcher("video") is video_fetcher

        assert registry.fetch("image", "a", 0) == b"img"
        assert registry.fetch("video", "b", 1) == b"vid"
