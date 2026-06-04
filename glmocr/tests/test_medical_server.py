"""medical_server 模块单元测试：覆盖参数校验、健康检查、解析主流程等"""

from __future__ import annotations

import struct
import zlib
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from glmocr.config import GlmOcrConfig
from glmocr.medical_server import create_app


# ── 辅助工具 ──────────────────────────────────────────────────────


def _make_minimal_png() -> bytes:
    """生成一个 1x1 白色 PNG 图片字节，用于 mock 文件拉取返回值"""
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\xff\xff\xff")
    idat = chunk(b"IDAT", raw)
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


# 预先生成，避免每次测试重复计算
_MINIMAL_PNG = _make_minimal_png()


def _create_app() -> "FastAPI":  # noqa: F821
    """创建测试用 FastAPI 应用实例"""
    config = GlmOcrConfig()
    return create_app(config)


def _valid_payload(**overrides) -> dict:
    """构造一个合法的 /v1/bills/parse 请求体"""
    payload = {
        "request_id": "test-req-001",
        "system": "PA_OCR",
        "regsno": "REGS001",
        "file_list": [
            {
                "page_count": 1,
                "file_type": "file_id",
                "file": "file-001",
            }
        ],
    }
    payload.update(overrides)
    return payload


# ── 测试用例 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_request_id():
    """POST 请求缺少 request_id，返回 HTTP 400 + code='400'"""
    app = _create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # request_id 为空字符串
        resp = await client.post(
            "/v1/bills/parse",
            json=_valid_payload(request_id=""),
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["code"] == "400"
        assert "request_id" in data["message"]


@pytest.mark.asyncio
async def test_missing_file_list():
    """POST 请求缺少 file_list，返回 HTTP 400"""
    app = _create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # file_list 为空列表
        resp = await client.post(
            "/v1/bills/parse",
            json=_valid_payload(file_list=[]),
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["code"] == "400"
        assert "file_list" in data["message"]


@pytest.mark.asyncio
async def test_unsupported_file_type():
    """file_type 不支持，返回 HTTP 400 + 'unsupported file_type'"""
    app = _create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/bills/parse",
            json=_valid_payload(
                file_list=[
                    {
                        "page_count": 1,
                        "file_type": "unsupported_type",
                        "file": "file-001",
                    }
                ]
            ),
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["code"] == "400"
        assert "unsupported file_type" in data["message"]


@pytest.mark.asyncio
async def test_health_ok():
    """GET /health 返回 200 + {'status': 'ok'}"""
    app = _create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "ok"}


@pytest.mark.asyncio
async def test_parse_success():
    """完整请求（mock 掉文件拉取和 OCR），返回 200 + code='200' + alg_request_time > 0"""
    app = _create_app()
    # mock 掉 registry.fetch，返回最小 PNG 字节
    with patch.object(app.state.registry, "fetch", return_value=_MINIMAL_PNG):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/bills/parse",
                json=_valid_payload(),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == "200"
            assert data["alg_request_time"] > 0
            assert data["request_id"] == "test-req-001"


@pytest.mark.asyncio
async def test_image_id_format():
    """验证 image_id 格式为 {file_id}_{seq}"""
    app = _create_app()

    # 使用两页的文件，验证 seq 从 1 开始递增
    captured_image_ids: list[str] = []

    original_fetch = app.state.registry.fetch

    def _mock_fetch(file_type: str, file_id: str, seq: int) -> bytes:
        # image_id 在路由中生成，这里只记录调用参数以供间接验证
        captured_image_ids.append(f"{file_id}_{seq}")
        return _MINIMAL_PNG

    with patch.object(app.state.registry, "fetch", side_effect=_mock_fetch):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/bills/parse",
                json=_valid_payload(
                    file_list=[
                        {
                            "page_count": 3,
                            "file_type": "file_id",
                            "file": "doc-abc",
                        }
                    ]
                ),
            )
            assert resp.status_code == 200
            data = resp.json()

            # fetch 被调用了 3 次，image_id 依次为 doc-abc_1, doc-abc_2, doc-abc_3
            assert captured_image_ids == ["doc-abc_1", "doc-abc_2", "doc-abc_3"]

            # 由于没有 ocr_client，所有票据 raw_markdown 为空，
            # 归并器会把它们放入 discarded_image（因为没有 medical_records）
            # 验证 discarded_image 中包含预期的 image_id
            assert "doc-abc_1" in data["discarded_image"]
            assert "doc-abc_2" in data["discarded_image"]
            assert "doc-abc_3" in data["discarded_image"]


@pytest.mark.asyncio
async def test_partial_download_failure():
    """部分图片下载失败时，失败的进入 discarded_image，成功的正常处理"""
    app = _create_app()

    call_count = 0

    def _mock_fetch_partial(file_type: str, file_id: str, seq: int) -> bytes:
        nonlocal call_count
        call_count += 1
        # 第一张成功，第二张失败
        if seq == 2:
            from glmocr.file_fetcher import FileFetchError

            raise FileFetchError(file_id, seq, "模拟下载失败")
        return _MINIMAL_PNG

    with patch.object(app.state.registry, "fetch", side_effect=_mock_fetch_partial):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/bills/parse",
                json=_valid_payload(
                    file_list=[
                        {
                            "page_count": 2,
                            "file_type": "file_id",
                            "file": "file-partial",
                        }
                    ]
                ),
            )
            assert resp.status_code == 200
            data = resp.json()

            # 第二张下载失败，应出现在 discarded_image 中
            assert "file-partial_2" in data["discarded_image"]

            # 第一张下载成功，但由于没有 medical_records，归并器也会把它放入 discarded_image
            # 所以 discarded_image 应包含两张
            assert "file-partial_1" in data["discarded_image"]


@pytest.mark.asyncio
async def test_alg_request_time_present():
    """响应中 alg_request_time 为 float 且 >= 0"""
    app = _create_app()
    with patch.object(app.state.registry, "fetch", return_value=_MINIMAL_PNG):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/bills/parse",
                json=_valid_payload(),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data["alg_request_time"], float)
            assert data["alg_request_time"] >= 0
