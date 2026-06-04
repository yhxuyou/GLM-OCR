"""医疗票据 FastAPI 服务：文件拉取 → OCR → 字段抽取 → 归并聚合"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from glmocr.config import (
    FileFetcherConfig,
    GlmOcrConfig,
    MedicalExtractorConfig,
    MedicalServerConfig,
    load_config,
)
from glmocr.file_fetcher import FileFetchError, FileFetcherRegistry, FileIdFetcher, UnsupportedFileType
from glmocr.medical_aggregator import (
    AggregationResult,
    CaseInfo,
    MedicalAggregator,
    MedicalRecordInput,
    PassThroughData,
)
from glmocr.medical_extractor import BillRecord, FieldExtractionError, MedicalFieldExtractor

logger = logging.getLogger(__name__)


# ── Pydantic 请求/响应模型 ──────────────────────────────────────────


class FileItem(BaseModel):
    """文件列表中的单个文件条目"""

    page_count: int
    file_type: str
    file: str


class MedicalRecordItem(BaseModel):
    """病历记录条目"""

    medical_id: str
    hospital_name: str = ""
    outpatientDate: str = ""
    startDate: str = ""
    endDate: str = ""


class PassThroughDataRequest(BaseModel):
    """透传数据"""

    medical_records: List[MedicalRecordItem] = []


class ParseRequest(BaseModel):
    """票据解析请求"""

    request_id: str
    system: str = "PA_OCR"
    regsno: str
    file_list: List[FileItem]
    pass_through_data: PassThroughDataRequest = Field(default_factory=PassThroughDataRequest)


class ParseResponse(BaseModel):
    """票据解析响应"""

    request_id: str
    code: str
    message: str
    alg_request_time: float = 0.0
    regsno: str = ""
    pass_through_data: dict = {}
    discarded_image: List[str] = []
    case_info: List[dict] = []


# ── 同步处理函数（在线程池中执行） ──────────────────────────────────


def _process_single_image(
    image_bytes: bytes,
    image_id: str,
    extractor: MedicalFieldExtractor,
    ocr_client,
) -> BillRecord:
    """处理单张图片：PIL 打开 → OCR → 字段抽取

    如果没有 ocr_client（测试/开发模式），生成一个空的 BillRecord。
    """
    if ocr_client is None:
        # 测试/开发模式：跳过 OCR，生成空记录
        return BillRecord(image_id=image_id, raw_markdown="")

    # PIL 打开图片（验证图片有效性）
    from PIL import Image

    try:
        Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        logger.warning("图片 %s 无法被 PIL 打开: %s，生成空记录", image_id, exc)
        return BillRecord(image_id=image_id, raw_markdown="")

    # 调用 OCR 获取 markdown
    try:
        markdown = ocr_client.process(image_bytes)
    except Exception as exc:
        logger.warning("图片 %s OCR 识别失败: %s，生成空记录", image_id, exc)
        return BillRecord(image_id=image_id, raw_markdown="")

    # 调用字段抽取
    try:
        return extractor.extract(image_id, markdown)
    except FieldExtractionError as exc:
        logger.warning("图片 %s 字段抽取失败: %s，生成空记录", image_id, exc)
        return BillRecord(image_id=image_id, raw_markdown=markdown)


# ── create_app 工厂函数 ─────────────────────────────────────────────


def create_app(config: GlmOcrConfig, preprocessor=None) -> FastAPI:
    """创建 FastAPI 应用实例

    Args:
        config: 全局配置对象
        preprocessor: 可选的 DocumentPreprocessor 工厂，传入则创建 PreprocessPool

    Returns:
        FastAPI 应用实例
    """
    app = FastAPI(title="Medical Bill OCR Server")

    server_cfg: MedicalServerConfig = config.medical_server
    fetcher_cfg: FileFetcherConfig = server_cfg.file_fetcher
    extractor_cfg: MedicalExtractorConfig = server_cfg.extractor

    # 初始化文件拉取器注册表
    registry = FileFetcherRegistry(
        supported_types=server_cfg.supported_file_types,
        fetcher_factory={
            "file_id": lambda: FileIdFetcher(
                base_url=fetcher_cfg.file_id_base_url,
                timeout=fetcher_cfg.timeout,
                retries=fetcher_cfg.retries,
                headers=fetcher_cfg.headers,
            ),
        },
    )

    # 初始化字段抽取器（ocr_client 暂时用 None 占位）
    extractor = MedicalFieldExtractor(
        ocr_client=None,
        prompt_template=extractor_cfg.prompt_template,
        retry_count=extractor_cfg.retry_count,
        temperature=extractor_cfg.temperature,
        retry_temperature=extractor_cfg.retry_temperature,
    )

    # 初始化归并器
    aggregator = MedicalAggregator(date_window_days=extractor_cfg.match_date_window_days)

    # 如果提供了 preprocessor，创建 PreprocessPool
    preprocess_pool = None
    if preprocessor is not None:
        from glmocr.preprocess_pool import PreprocessPool

        preprocess_pool = PreprocessPool(preprocessor, config)

    # 将组件存到 app.state
    app.state.config = config
    app.state.registry = registry
    app.state.extractor = extractor
    app.state.aggregator = aggregator
    app.state.preprocess_pool = preprocess_pool
    app.state.ocr_client = None  # 由调用者后续注入

    # ── 全局异常处理 ────────────────────────────────────────────────

    @app.exception_handler(UnsupportedFileType)
    async def unsupported_file_type_handler(request: Request, exc: UnsupportedFileType):
        """不支持的文件类型 → 400"""
        body = await _safe_parse_request(request)
        request_id = body.get("request_id", "") if body else ""
        return JSONResponse(
            status_code=400,
            content={
                "code": "400",
                "message": f"unsupported file_type: {exc.file_type}",
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """未捕获异常 → 500"""
        body = await _safe_parse_request(request)
        request_id = body.get("request_id", "") if body else ""
        logger.exception("未捕获异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "code": "500",
                "message": "internal error",
                "request_id": request_id,
            },
        )

    # ── 路由 ────────────────────────────────────────────────────────

    @app.post("/v1/bills/parse", response_model=ParseResponse)
    async def parse_bills(request: Request):
        """医疗票据解析主路由"""
        # 1. 解析并校验入参
        try:
            body = await request.json()
            parse_req = ParseRequest.model_validate(body)
        except Exception as exc:
            # Pydantic ValidationError 等 → 400
            return JSONResponse(
                status_code=400,
                content={
                    "code": "400",
                    "message": f"validation error: {exc}",
                    "request_id": "",
                },
            )

        # 2. 额外校验
        if not parse_req.request_id:
            return JSONResponse(
                status_code=400,
                content={
                    "code": "400",
                    "message": "validation error: request_id is required",
                    "request_id": "",
                },
            )
        if not parse_req.file_list:
            return JSONResponse(
                status_code=400,
                content={
                    "code": "400",
                    "message": "validation error: file_list is required",
                    "request_id": parse_req.request_id,
                },
            )
        for file_item in parse_req.file_list:
            if file_item.file_type not in server_cfg.supported_file_types:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "400",
                        "message": f"unsupported file_type: {file_item.file_type}",
                        "request_id": parse_req.request_id,
                    },
                )

        # 3. 记录开始时间
        start_time = time.time()

        # 4. 遍历 file_list，下载图片
        downloaded_images: list[tuple[str, bytes]] = []  # (image_id, image_bytes)
        discarded_image: list[str] = []

        for file_item in parse_req.file_list:
            for seq in range(1, file_item.page_count + 1):
                image_id = f"{file_item.file}_{seq}"
                try:
                    image_bytes = await asyncio.to_thread(
                        registry.fetch, file_item.file_type, file_item.file, seq
                    )
                    downloaded_images.append((image_id, image_bytes))
                except (FileFetchError, UnsupportedFileType) as exc:
                    logger.warning("图片 %s 下载失败: %s", image_id, exc)
                    discarded_image.append(image_id)
                except Exception as exc:
                    logger.warning("图片 %s 下载异常: %s", image_id, exc)
                    discarded_image.append(image_id)

        # 5. 对成功下载的图片，逐张调用预处理+OCR+字段抽取
        ocr_client = app.state.ocr_client
        bills: list[BillRecord] = []

        for image_id, image_bytes in downloaded_images:
            try:
                bill_record = await asyncio.to_thread(
                    _process_single_image,
                    image_bytes,
                    image_id,
                    extractor,
                    ocr_client,
                )
                bills.append(bill_record)
            except Exception as exc:
                logger.warning("图片 %s 处理失败，加入 discarded: %s", image_id, exc)
                discarded_image.append(image_id)

        # 6. 构建病历记录列表，调用归并器
        medical_records = [
            MedicalRecordInput(
                medical_id=mr.medical_id,
                hospital_name=mr.hospital_name,
                outpatientDate=mr.outpatientDate,
                startDate=mr.startDate,
                endDate=mr.endDate,
            )
            for mr in parse_req.pass_through_data.medical_records
        ]

        aggregation_result: AggregationResult = await asyncio.to_thread(
            aggregator.aggregate, bills, medical_records
        )

        # 合并归并器返回的 discarded_image
        discarded_image.extend(aggregation_result.discarded_image)

        # 7. 计算耗时
        alg_request_time = time.time() - start_time

        # 8. 组装响应
        response = ParseResponse(
            request_id=parse_req.request_id,
            code="200",
            message="success",
            alg_request_time=round(alg_request_time, 3),
            regsno=parse_req.regsno,
            pass_through_data=parse_req.pass_through_data.model_dump(),
            discarded_image=discarded_image,
            case_info=[ci.model_dump() for ci in aggregation_result.case_info],
        )
        return response

    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {"status": "ok"}

    return app


async def _safe_parse_request(request: Request) -> Optional[dict]:
    """安全地从请求体中解析 JSON，失败返回 None"""
    try:
        return await request.json()
    except Exception:
        return None


# ── CLI 入口 ────────────────────────────────────────────────────────


def main():
    """命令行入口：启动医疗票据 OCR 服务"""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Medical Bill OCR Server")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--host", type=str, default=None, help="监听地址")
    parser.add_argument("--port", type=int, default=None, help="监听端口")
    args = parser.parse_args()

    config = load_config(args.config)
    host = args.host or config.medical_server.host
    port = args.port or config.medical_server.port

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
