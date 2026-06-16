"""Pipeline 微服务 FastAPI 入口

对外暴露统一的文档处理接口，内部串联：
  预处理 → GLM-OCR 异步识别 → 后处理
"""

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import config, PipelineConfig
from .service import PipelineService, ServiceClientError


# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── 请求/响应模型 ────────────────────────────────────────────────

class ProcessResponse(BaseModel):
    """端到端处理响应"""
    task_id: str = Field(..., description="任务 ID")
    status: str = Field(..., description="状态: success / failed")
    elapsed: float = Field(..., description="耗时（秒）")
    ocr_result: Optional[Dict[str, Any]] = Field(None, description="OCR 结果")
    postprocess: Optional[Dict[str, Any]] = Field(None, description="后处理结果")
    error: Optional[str] = Field(None, description="错误信息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    pipeline: str = Field("ok", description="Pipeline 自身状态")
    preprocess: bool = Field(..., description="预处理服务是否可用")
    glmocr: bool = Field(..., description="GLM-OCR 服务是否可用")
    postprocess: bool = Field(..., description="后处理服务是否可用")


class Base64ProcessRequest(BaseModel):
    """Base64 输入的请求"""
    image_b64: str = Field(..., description="base64 编码图像")
    filename: str = Field("image.jpg", description="文件名")
    target_format: Optional[str] = Field(None, description="后处理目标格式")
    timeout: Optional[float] = Field(None, description="OCR 超时时间（秒）")


# ── App 创建 ─────────────────────────────────────────────────────

app = FastAPI(
    title="Document Pipeline Service",
    description="端到端文档处理流水线：预处理 → GLM-OCR → 后处理",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 Pipeline 实例
pipeline_service: Optional[PipelineService] = None


def get_pipeline() -> PipelineService:
    global pipeline_service
    if pipeline_service is None:
        pipeline_service = PipelineService()
    return pipeline_service


@app.on_event("startup")
async def startup():
    """启动时初始化"""
    global pipeline_service
    pipeline_service = PipelineService(config)
    logger.info(f"Pipeline 服务启动: {config.host}:{config.port}")
    health = await pipeline_service.health_check()
    logger.info(f"依赖服务健康状态: {health}")


# ── API 端点 ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查，返回所有依赖服务状态"""
    svc = get_pipeline()
    deps = await svc.health_check()
    return HealthResponse(**deps)


@app.post("/process", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(..., description="输入文档图像"),
    target_format: Optional[str] = Form(None, description="后处理目标格式"),
    timeout: Optional[float] = Form(None, description="OCR 超时时间（秒）"),
):
    """端到端文档处理

    上传文件 → 预处理 → GLM-OCR 异步识别 → 后处理 → 返回结果
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件")

    # 写入临时文件
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        svc = get_pipeline()
        result = await svc.process(
            file_path=tmp_path,
            target_format=target_format,
            timeout=timeout,
        )
        return ProcessResponse(**result)
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/process/base64", response_model=ProcessResponse)
async def process_base64_image(request: Base64ProcessRequest):
    """跳过预处理的端到端处理（输入 base64 图像）

    直接进行 GLM-OCR + 后处理。
    """
    try:
        svc = get_pipeline()
        result = await svc.process_base64(
            image_b64=request.image_b64,
            filename=request.filename,
            target_format=request.target_format,
            timeout=request.timeout,
        )
        return ProcessResponse(**result)
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "document-pipeline",
        "version": "1.0.0",
        "endpoints": {
            "POST /process": "端到端处理（上传文件）",
            "POST /process/base64": "端到端处理（base64 输入）",
            "GET /health": "健康检查",
        },
    }


# ── 入口函数 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline Server")
    parser.add_argument("--host", type=str, default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=2, help="Uvicorn worker 数量")
    args = parser.parse_args()

    host = args.host or config.host
    port = args.port or config.port

    logger.info(f"启动 Pipeline 服务: {host}:{port} (workers={args.workers})")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=logging.INFO.lower(),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
