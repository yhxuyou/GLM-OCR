"""
GLM-OCR FastAPI高并发服务
"""
import os
import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import get_settings
from .logger import logger
from .task_manager import get_task_manager, TaskStatus
from .ocr_processor import get_ocr_processor


# Pydantic模型
class OCRRequest(BaseModel):
    """OCR请求模型（JSON方式）"""
    image_url: Optional[str] = None
    save_layout_visualization: bool = False


class OCRResponse(BaseModel):
    """OCR响应模型"""
    success: bool
    task_id: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    queue_stats: Optional[dict] = None
    ocr_stats: Optional[dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting GLM-OCR service...")
    
    try:
        # 初始化OCR处理器
        ocr_processor = get_ocr_processor()
        ocr_processor.initialize()
        logger.info("OCR processor initialized")
        
        # 初始化任务管理器
        task_manager = get_task_manager()
        await task_manager.initialize()
        logger.info("Task manager initialized")
        
        logger.info("GLM-OCR service started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}", exc_info=True)
        raise
    
    finally:
        # 关闭资源
        logger.info("Shutting down GLM-OCR service...")
        
        try:
            task_manager = get_task_manager()
            await task_manager.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down task manager: {e}", exc_info=True)
        
        try:
            ocr_processor = get_ocr_processor()
            ocr_processor.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down OCR processor: {e}", exc_info=True)
        
        logger.info("GLM-OCR service shutdown complete")


# 创建FastAPI应用
settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="High-concurrency GLM-OCR service with async task processing",
    lifespan=lifespan
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


def get_client_id(request: Request) -> str:
    """获取客户端ID（用于速率限制）"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def validate_file_extension(filename: str) -> bool:
    """验证文件扩展名"""
    ext = Path(filename).suffix.lower().lstrip('.')
    return ext in settings.ALLOWED_EXTENSIONS


def get_file_extension(filename: str) -> str:
    """获取文件扩展名"""
    return Path(filename).suffix.lower().lstrip('.') or 'png'


@app.get("/", response_model=HealthResponse)
async def root():
    """根路径"""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    ocr_processor = get_ocr_processor()
    task_manager = get_task_manager()
    
    return HealthResponse(
        status="healthy" if ocr_processor.is_healthy() else "unhealthy",
        version=settings.APP_VERSION,
        queue_stats=await task_manager.get_queue_stats(),
        ocr_stats=ocr_processor.get_queue_stats()
    )


@app.post("/api/v1/ocr/sync", response_model=OCRResponse)
async def ocr_sync(
    request: Request,
    file: UploadFile = File(...),
    save_layout_visualization: bool = Form(False)
):
    """
    同步OCR处理（不推荐用于高并发场景）
    直接返回结果，不使用任务队列
    """
    logger.info(f"Received sync OCR request for file: {file.filename}")
    
    # 验证文件
    if not validate_file_extension(file.filename or ''):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # 读取文件内容
    file_bytes = await file.read()
    
    # 检查文件大小
    if len(file_bytes) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB"
        )
    
    try:
        # 直接处理（不使用任务队列）
        ocr_processor = get_ocr_processor()
        file_ext = get_file_extension(file.filename or '')
        
        result = ocr_processor.process_bytes(
            file_bytes,
            file_ext,
            save_layout_visualization=save_layout_visualization
        )
        
        return OCRResponse(
            success=True,
            result=result
        )
        
    except Exception as e:
        logger.error(f"Sync OCR error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/v1/ocr/async", response_model=OCRResponse)
async def ocr_async(
    request: Request,
    file: UploadFile = File(...),
    save_layout_visualization: bool = Form(False)
):
    """
    异步OCR处理（推荐用于高并发场景）
    创建任务并立即返回task_id
    """
    logger.info(f"Received async OCR request for file: {file.filename}")
    
    # 验证文件
    if not validate_file_extension(file.filename or ''):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # 读取文件内容
    file_bytes = await file.read()
    
    # 检查文件大小
    if len(file_bytes) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB"
        )
    
    # 保存临时文件
    file_ext = get_file_extension(file.filename or '')
    temp_filename = f"{uuid.uuid4()}.{file_ext}"
    temp_path = settings.UPLOAD_DIR / temp_filename
    
    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        
        # 创建任务处理器
        def process_task():
            ocr_processor = get_ocr_processor()
            try:
                return ocr_processor.process_file(
                    temp_path,
                    save_layout_visualization=save_layout_visualization
                )
            finally:
                # 清理临时文件
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
        
        # 创建任务
        task_manager = get_task_manager()
        client_id = get_client_id(request)
        task_id = await task_manager.create_task(
            process_task,
            metadata={
                "filename": file.filename,
                "file_size": len(file_bytes)
            },
            client_id=client_id
        )
        
        return OCRResponse(
            success=True,
            task_id=task_id
        )
        
    except RuntimeError as e:
        if "Rate limit exceeded" in str(e):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        elif "Task queue is full" in str(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service busy, please try again later"
            )
        raise
    except Exception as e:
        # 清理临时文件
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        
        logger.error(f"Async OCR error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """获取任务状态和结果"""
    task_manager = get_task_manager()
    task = await task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    return TaskResponse(
        task_id=task.task_id,
        status=task.status.value,
        result=task.result,
        error=task.error
    )


@app.get("/api/v1/tasks")
async def list_tasks(status: Optional[str] = None):
    """列出所有任务"""
    task_manager = get_task_manager()
    
    try:
        task_status = TaskStatus(status) if status else None
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status. Allowed: {[s.value for s in TaskStatus]}"
        )
    
    tasks = await task_manager.get_all_tasks(status=task_status)
    
    return {
        "tasks": [t.to_dict() for t in tasks],
        "count": len(tasks)
    }


@app.post("/api/v1/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务"""
    task_manager = get_task_manager()
    success = await task_manager.cancel_task(task_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found or cannot be cancelled"
        )
    
    return {"success": True, "task_id": task_id}


@app.get("/api/v1/stats")
async def get_stats():
    """获取服务统计信息"""
    task_manager = get_task_manager()
    ocr_processor = get_ocr_processor()
    
    queue_stats = await task_manager.get_queue_stats()
    ocr_stats = ocr_processor.get_queue_stats()
    
    return {
        "queue": queue_stats,
        "ocr": ocr_stats
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "deploy.fastapi.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1  # 生产环境建议使用单个worker，因为GLM-OCR内部已有并发处理
    )
