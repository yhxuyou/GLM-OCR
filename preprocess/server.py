"""
预处理微服务 FastAPI 入口

提供 HTTP API 接口，支持：
- 单步骤调用（/detect, /orient, /dewarp）
- 完整流水线调用（/preprocess）
- 健康检查（/health）
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from preprocess.config import load_config, PreprocessConfig
from preprocess.pipeline import PreprocessPipeline
from preprocess.utils.logging import get_logger, configure_logging

logger = get_logger(__name__)

# 全局应用实例
app = FastAPI(
    title="预处理微服务",
    description="文档预处理服务：文档检测、方向矫正、扭曲矫正",
    version="1.0.0"
)

# 全局配置和流水线实例
_config: Optional[PreprocessConfig] = None
_pipeline: Optional[PreprocessPipeline] = None


def _get_pipeline() -> PreprocessPipeline:
    """获取流水线实例"""
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("流水线未初始化")
    return _pipeline


def _image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """将 PIL 图像转换为 base64 字符串
    
    Args:
        image: PIL 图像
        format: 图像格式（PNG, JPEG 等）
        
    Returns:
        base64 编码的字符串
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _base64_to_image(base64_str: str) -> Image.Image:
    """将 base64 字符串转换为 PIL 图像
    
    Args:
        base64_str: base64 编码的字符串
        
    Returns:
        PIL 图像
    """
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))


@app.on_event("startup")
async def startup_event():
    """应用启动事件：加载模型"""
    global _config, _pipeline
    
    logger.info("预处理微服务启动中...")
    
    # 加载配置
    _config = load_config()
    configure_logging(level=_config.logging.level)
    
    # 初始化流水线
    _pipeline = PreprocessPipeline(_config)
    _pipeline.load_models()
    
    # 模型预热
    _pipeline.warmup()
    
    logger.info(f"预处理微服务启动完成，监听 {_config.server.host}:{_config.server.port}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件：卸载模型"""
    global _pipeline
    
    logger.info("预处理微服务关闭中...")
    
    if _pipeline:
        _pipeline.unload_models()
    
    logger.info("预处理微服务已关闭")


@app.get("/health")
async def health_check():
    """健康检查端点
    
    Returns:
        服务状态信息
    """
    pipeline = _get_pipeline()
    
    return {
        "status": "healthy",
        "models_loaded": pipeline._loaded,
        "doc_detector": pipeline.doc_detector.is_loaded() if pipeline.doc_detector else False,
        "orientation_corrector": pipeline.orientation_corrector.is_loaded() if pipeline.orientation_corrector else False,
        "dewarp_corrector": pipeline.dewarp_corrector.is_loaded() if pipeline.dewarp_corrector else False
    }


@app.post("/detect")
async def detect_document(
    file: UploadFile = File(..., description="输入图像文件")
):
    """文档检测端点
    
    检测图像中的文档区域边界
    
    Args:
        file: 上传的图像文件
        
    Returns:
        检测结果，包含边界框和置信度
    """
    pipeline = _get_pipeline()
    
    if not pipeline.doc_detector or not pipeline.doc_detector.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="文档检测模型未加载"
        )
    
    try:
        # 读取图像
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # 执行检测
        result = pipeline.doc_detector.predict(image)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"文档检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orient")
async def correct_orientation(
    file: UploadFile = File(..., description="输入图像文件"),
    apply_rotation: bool = Form(False, description="是否应用旋转变换")
):
    """方向矫正端点
    
    检测并矫正文档的旋转方向
    
    Args:
        file: 上传的图像文件
        apply_rotation: 是否返回旋转后的图像
        
    Returns:
        矫正结果，包含旋转角度和矫正后的图像
    """
    pipeline = _get_pipeline()
    
    if not pipeline.orientation_corrector or not pipeline.orientation_corrector.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="方向矫正模型未加载"
        )
    
    try:
        # 读取图像
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # 执行方向检测
        result = pipeline.orientation_corrector.predict(image)
        
        # 如果需要应用旋转
        if apply_rotation:
            from preprocess.models.orientation_corrector import OrientationCorrectorLoader
            angle = result.get("angle", 0)
            if angle != 0:
                corrected_image = OrientationCorrectorLoader.apply_rotation(image, angle)
                result["corrected_image"] = _image_to_base64(corrected_image)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"方向矫正失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dewarp")
async def correct_dewarp(
    file: UploadFile = File(..., description="输入图像文件")
):
    """扭曲矫正端点
    
    矫正文档的扭曲变形
    
    Args:
        file: 上传的图像文件
        
    Returns:
        矫正结果，包含矫正后的图像
    """
    pipeline = _get_pipeline()
    
    if not pipeline.dewarp_corrector or not pipeline.dewarp_corrector.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="扭曲矫正模型未加载"
        )
    
    try:
        # 读取图像
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # 执行扭曲矫正
        result = pipeline.dewarp_corrector.predict(image)
        
        # 将矫正后的图像转换为 base64
        if "corrected_image" in result:
            result["corrected_image"] = _image_to_base64(result["corrected_image"])
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"扭曲矫正失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess")
async def preprocess_full(
    file: UploadFile = File(..., description="输入图像文件"),
    return_intermediate: bool = Form(False, description="是否返回中间结果")
):
    """完整预处理流水线端点
    
    串行执行：文档检测 → 方向矫正 → 扭曲矫正
    
    Args:
        file: 上传的图像文件
        return_intermediate: 是否返回每个步骤的中间结果
        
    Returns:
        预处理结果，包含最终图像和可选的中间结果
    """
    pipeline = _get_pipeline()
    
    try:
        # 读取图像
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # 执行完整流水线
        result = pipeline.process(image, return_intermediate=return_intermediate)
        
        # 将最终图像转换为 base64
        if "final_image" in result:
            result["final_image"] = _image_to_base64(result["final_image"])
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"预处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess/batch")
async def preprocess_batch(
    files: list[UploadFile] = File(..., description="输入图像文件列表"),
    return_intermediate: bool = Form(False, description="是否返回中间结果")
):
    """批量预处理端点
    
    Args:
        files: 上传的图像文件列表
        return_intermediate: 是否返回每个步骤的中间结果
        
    Returns:
        预处理结果列表
    """
    pipeline = _get_pipeline()
    
    try:
        # 读取所有图像
        images = []
        for file in files:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            images.append(image)
        
        # 执行批量预处理
        results = pipeline.process_batch(images, return_intermediate=return_intermediate)
        
        # 将所有最终图像转换为 base64
        for result in results:
            if "final_image" in result:
                result["final_image"] = _image_to_base64(result["final_image"])
        
        return JSONResponse(content={"results": results})
    
    except Exception as e:
        logger.error(f"批量预处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """主函数：启动服务"""
    config = load_config()
    configure_logging(level=config.logging.level)
    
    logger.info(f"启动预处理微服务: {config.server.host}:{config.server.port}")
    
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower()
    )


if __name__ == "__main__":
    main()
