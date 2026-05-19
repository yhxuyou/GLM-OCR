"""
Flask API 路由模块
"""
import json
import uuid
from pathlib import Path
from datetime import datetime, UTC
from flask import Blueprint, request, jsonify, send_file, Response
from mimetypes import guess_type

from app.schemas.response import ApiResponse
from app.core.task_manager import get_task_manager
from app.utils.upload_file_manager import file_upload_handler
from app.utils.config import settings

router = Blueprint("api", __name__, url_prefix="/api/v1")


@router.route("/tasks/upload", methods=["POST"])
def submit_task():
    """
    提交新任务
    """
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({
                "success": False,
                "error_code": "NO_FILE",
                "message": "No file provided"
            }), 400

        processing_mode = request.form.get("processing_mode", "pipeline")
        priority = int(request.form.get("priority", 2))
        custom_url = request.form.get("custom_url")
        output_format = request.form.get("output_format", "markdown")

        document_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())

        parsed_ocr_config = None
        if custom_url is not None:
            parsed_ocr_config = {"custom_url": custom_url}

        save_dir = str(Path(settings.OUTPUT_DIR) / task_id)
        saved_path = save_file_to_path(file, file.filename, save_dir)
        saved_path_obj = Path(saved_path)
        file_size = saved_path_obj.stat().st_size
        file_type = saved_path_obj.suffix.lstrip(".").lower()

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task_manager = get_task_manager()
            loop.run_until_complete(task_manager.submit_task(
                task_id=task_id,
                document_id=document_id,
                original_filename=file.filename,
                file_type=file_type,
                file_size=file_size,
                file_path=str(saved_path_obj),
                processing_mode=processing_mode,
                priority=priority,
                ocr_config=parsed_ocr_config,
                output_format=output_format,
            ))
        finally:
            loop.close()

        return jsonify({
            "success": True,
            "data": {
                "task_id": task_id,
                "document_id": document_id,
                "status": "pending",
                "processing_mode": processing_mode,
                "priority": priority,
                "created_at": datetime.now(UTC).isoformat(),
            },
            "message": "Task submitted successfully",
        }), 201

    except Exception as e:
        return jsonify({
            "success": False,
            "error_code": "SUBMIT_ERROR",
            "message": f"Failed to submit task: {str(e)}"
        }), 500


def save_file_to_path(file_storage, filename, upload_dir):
    """同步保存文件"""
    from werkzeug.utils import secure_filename
    safe_name = secure_filename(filename or file_storage.filename or "upload")
    dest_path = Path(upload_dir) / safe_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    file_storage.save(str(dest_path))
    return str(dest_path)


@router.route("/tasks/file", methods=["GET"])
def read_file():
    """读取指定路径的文件内容"""
    try:
        path = request.args.get("path")
        if not path:
            return jsonify({
                "success": False,
                "error_code": "NO_PATH",
                "message": "No path provided"
            }), 400

        file_path = Path(path)
        if not file_path.exists():
            return jsonify({
                "success": False,
                "error_code": "FILE_NOT_FOUND",
                "message": f"File not found: {path}"
            }), 404

        if not file_path.is_file():
            return jsonify({
                "success": False,
                "error_code": "NOT_FILE",
                "message": f"Path is not a file: {path}"
            }), 400

        mime_type, _ = guess_type(file_path.name)
        if mime_type is None:
            mime_type = "application/octet-stream"

        if mime_type.startswith("image/"):
            return send_file(
                file_path,
                mimetype=mime_type,
                as_attachment=False,
                download_name=file_path.name
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            content = "(binary file)"

        return jsonify({
            "success": True,
            "data": {
                "path": str(file_path.absolute()),
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "mime_type": mime_type,
                "content": content,
            },
            "message": "File read successfully",
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error_code": "READ_ERROR",
            "message": f"Failed to read file: {str(e)}"
        }), 500


@router.route("/tasks/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """获取任务状态"""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task_manager = get_task_manager()
            task_info = loop.run_until_complete(task_manager.get_task_status(task_id))
        finally:
            loop.close()

        if not task_info:
            return jsonify({
                "success": False,
                "error_code": "TASK_NOT_FOUND",
                "message": f"Task not found: {task_id}"
            }), 404

        result_file_path = task_info.get("result_file_path")
        result_data = None
        if result_file_path:
            try:
                result_path = Path(result_file_path)
                if result_path.exists():
                    with open(result_path, "r", encoding="utf-8") as f:
                        result_data = json.load(f)
            except Exception:
                pass

        response_data = {
            "task_id": task_info.get("task_id"),
            "document_id": task_info.get("document_id"),
            "status": task_info.get("status"),
            "progress": task_info.get("progress"),
            "current_step": task_info.get("current_step"),
            "created_at": task_info.get("created_at").isoformat() if task_info.get("created_at") else None,
            "started_at": task_info.get("started_at").isoformat() if task_info.get("started_at") else None,
            "completed_at": task_info.get("completed_at").isoformat() if task_info.get("completed_at") else None,
            "error_message": task_info.get("error_message"),
            "processing_mode": task_info.get("processing_mode"),
            "priority": task_info.get("priority"),
            "retry_count": task_info.get("retry_count"),
            "worker_id": task_info.get("worker_id"),
        }

        if result_data:
            response_data["metadata"] = result_data.get("metadata")
            response_data["full_markdown"] = result_data.get("full_markdown")
            response_data["layout"] = result_data.get("layout")

        return jsonify({
            "success": True,
            "data": response_data,
            "message": "Task status retrieved successfully",
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error_code": "STATUS_ERROR",
            "message": f"Failed to get task status: {str(e)}"
        }), 500


@router.route("/tasks/<task_id>", methods=["DELETE"])
def cancel_task(task_id):
    """取消任务"""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task_manager = get_task_manager()
            success = loop.run_until_complete(task_manager.cancel_task(task_id))
        finally:
            loop.close()

        if not success:
            return jsonify({
                "success": False,
                "error_code": "CANCEL_FAILED",
                "message": f"Task not found or cannot be cancelled: {task_id}"
            }), 404

        return jsonify({
            "success": True,
            "data": {
                "task_id": task_id,
                "status": "cancelled",
            },
            "message": "Task cancelled successfully",
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error_code": "CANCEL_ERROR",
            "message": f"Failed to cancel task: {str(e)}"
        }), 500


@router.route("/tasks", methods=["GET"])
def list_tasks():
    """列出任务"""
    try:
        status_filter = request.args.get("status")
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task_manager = get_task_manager()
            tasks = loop.run_until_complete(
                task_manager.list_tasks(status=status_filter, limit=limit, offset=offset)
            )
        finally:
            loop.close()

        return jsonify({
            "success": True,
            "data": {
                "tasks": tasks,
                "total": len(tasks),
                "limit": limit,
                "offset": offset,
            },
            "message": "Tasks retrieved successfully",
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error_code": "LIST_ERROR",
            "message": f"Failed to list tasks: {str(e)}"
        }), 500


@router.route("/system/metrics", methods=["GET"])
def get_metrics():
    """获取系统指标"""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task_manager = get_task_manager()
            metrics = loop.run_until_complete(task_manager.get_metrics())
        finally:
            loop.close()

        return jsonify({
            "success": True,
            "data": metrics,
            "message": "Metrics retrieved successfully",
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error_code": "METRICS_ERROR",
            "message": f"Failed to get metrics: {str(e)}"
        }), 500


@router.route("/system/health", methods=["GET"])
def system_health():
    """系统健康检查"""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task_manager = get_task_manager()
            metrics = loop.run_until_complete(task_manager.get_metrics())
        finally:
            loop.close()

        return jsonify({
            "success": True,
            "data": {
                "status": "healthy",
                "task_manager_running": task_manager.is_running,
                "workers_count": len(task_manager.workers),
                "active_workers": metrics.get("workers", {}).get("active", 0),
                "version": settings.APP_VERSION
            },
            "message": "System is healthy"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "data": {
                "status": "unhealthy",
                "error": str(e)
            },
            "message": "System health check failed"
        }), 500
