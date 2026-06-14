"""后处理微服务 FastAPI 应用

提供后处理 HTTP 接口，支持单个和批量处理。
"""

import logging
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import PostprocessConfig, config
from .processor import ResultProcessor


# 配置日志
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 请求/响应模型
class PostprocessRequest(BaseModel):
    """后处理请求模型"""
    data: Union[Dict[str, Any], List[Any], str] = Field(
        ...,
        description="待处理的数据"
    )
    target_format: Optional[str] = Field(
        default=None,
        description="目标格式（json/markdown/html）"
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="额外的处理选项"
    )


class PostprocessResponse(BaseModel):
    """后处理响应模型"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Any] = Field(None, description="处理后的数据")
    error: Optional[str] = Field(None, description="错误信息")


class BatchPostprocessRequest(BaseModel):
    """批量后处理请求模型"""
    items: List[PostprocessRequest] = Field(
        ...,
        description="批量请求列表"
    )


class BatchPostprocessResponse(BaseModel):
    """批量后处理响应模型"""
    success: bool = Field(..., description="是否成功")
    results: List[PostprocessResponse] = Field(
        ...,
        description="处理结果列表"
    )
    total: int = Field(..., description="总请求数")
    succeeded: int = Field(..., description="成功数")
    failed: int = Field(..., description="失败数")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    service: str = Field(..., description="服务名称")


# 全局处理器实例
processor: Optional[ResultProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    global processor
    logger.info("后处理服务启动中...")
    processor = ResultProcessor(config)
    logger.info("后处理服务启动完成")
    
    yield
    
    # 关闭时清理
    logger.info("后处理服务关闭中...")
    processor = None
    logger.info("后处理服务已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="OCR 结果后处理微服务",
    lifespan=lifespan
)


# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点
    
    Returns:
        健康检查响应
    """
    return HealthResponse(
        status="healthy",
        version=config.app_version,
        service=config.app_name
    )


@app.post("/postprocess", response_model=PostprocessResponse)
async def postprocess(request: PostprocessRequest):
    """后处理端点
    
    对单个数据进行后处理。
    
    Args:
        request: 后处理请求
        
    Returns:
        后处理响应
    """
    if processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务未就绪"
        )
    
    try:
        # 处理数据
        result = processor.process(
            request.data,
            target_format=request.target_format,
            **(request.options or {})
        )
        
        return PostprocessResponse(
            success=True,
            data=result,
            error=None
        )
    except Exception as e:
        logger.error(f"后处理失败: {e}", exc_info=True)
        return PostprocessResponse(
            success=False,
            data=None,
            error=str(e)
        )


@app.post("/postprocess/batch", response_model=BatchPostprocessResponse)
async def postprocess_batch(request: BatchPostprocessRequest):
    """批量后处理端点
    
    对多个数据进行批量后处理。
    
    Args:
        request: 批量后处理请求
        
    Returns:
        批量后处理响应
    """
    if processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务未就绪"
        )
    
    results = []
    succeeded = 0
    failed = 0
    
    for item in request.items:
        try:
            result = processor.process(
                item.data,
                target_format=item.target_format,
                **(item.options or {})
            )
            results.append(PostprocessResponse(
                success=True,
                data=result,
                error=None
            ))
            succeeded += 1
        except Exception as e:
            logger.error(f"批量后处理失败: {e}", exc_info=True)
            results.append(PostprocessResponse(
                success=False,
                data=None,
                error=str(e)
            ))
            failed += 1
    
    return BatchPostprocessResponse(
        success=True,
        results=results,
        total=len(request.items),
        succeeded=succeeded,
        failed=failed
    )


@app.get("/")
async def root():
    """根路径
    
    Returns:
        服务基本信息
    """
    return {
        "name": config.app_name,
        "version": config.app_version,
        "status": "running",
        "docs": "/docs"
    }


def create_app(config_instance: Optional[PostprocessConfig] = None) -> FastAPI:
    """创建 FastAPI 应用实例
    
    允许传入自定义配置创建应用实例。
    
    Args:
        config_instance: 自定义配置实例
        
    Returns:
        FastAPI 应用实例
    """
    global config, processor
    
    if config_instance:
        config = config_instance
    
    # 重新初始化处理器
    processor = ResultProcessor(config)
    
    return app


def run_server():
    """运行服务"""
    import uvicorn
    
    uvicorn.run(
        "postprocess.server:app",
        host=config.host,
        port=config.port,
        reload=config.debug
    )


if __name__ == "__main__":
    run_server()
