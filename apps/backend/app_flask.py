"""
Flask应用入口
"""
import asyncio
from flask import Flask, jsonify
from flask_cors import CORS

from app_flask_api import router as api_router
from app.core.task_manager import init_task_system, shutdown_task_system
from app.db.database import init_db, close_db
from app.utils.logger import logger
from app.utils.config import settings


def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    app.register_blueprint(api_router)

    @app.route("/")
    def root():
        """根路径"""
        return jsonify({
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running"
        })

    @app.route("/health")
    def health_check():
        """健康检查"""
        return jsonify({
            "status": "healthy",
            "version": settings.APP_VERSION
        })

    return app


async def startup():
    """启动时初始化"""
    logger.info("Application starting up...")

    await init_db()
    await init_task_system()

    logger.info("Application startup complete")


async def shutdown():
    """关闭时清理"""
    logger.info("Application shutting down...")

    await shutdown_task_system()
    await close_db()

    logger.info("Application shutdown complete")


def run_flask_app():
    """运行Flask应用（带异步初始化）"""
    app = create_app()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(startup())

        app.run(
            host=settings.HOST,
            port=settings.PORT,
            debug=settings.DEBUG,
            use_reloader=False
        )
    finally:
        loop.run_until_complete(shutdown())
        loop.close()


if __name__ == "__main__":
    run_flask_app()
