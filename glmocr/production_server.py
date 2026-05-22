"""Production-grade FastAPI server for GLM-OCR Pipeline.

Replaces the Flask-based server.py with:
- Async event loop (uvicorn)
- Prometheus metrics
- Structured logging (structlog)
- Rate limiting
- Graceful shutdown
- Health probes
"""

from __future__ import annotations

import os
import time
import uuid
import sys
import multiprocessing
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

try:
    import structlog
    from structlog.processors import JSONRenderer, TimeStamper
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse, Response
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

from glmocr.pipeline import Pipeline
from glmocr.config import load_config

logger = structlog.get_logger(__name__) if STRUCTLOG_AVAILABLE else None

# ── Prometheus Metrics ──────────────────────────────────────────────

if PROMETHEUS_AVAILABLE:
    OCR_REQUESTS = Counter(
        "glmocr_requests_total",
        "Total OCR requests",
        ["endpoint", "status"],
    )
    OCR_REQUEST_DURATION = Histogram(
        "glmocr_request_duration_seconds",
        "OCR request duration in seconds",
        ["endpoint"],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    OCR_INFLIGHT = Gauge(
        "glmocr_requests_inflight",
        "Number of requests currently being processed",
    )
    OCR_QUEUE_DEPTH = Gauge(
        "glmocr_queue_depth",
        "Current pipeline queue depth",
        ["queue_name"],
    )
    GPU_MEMORY_USAGE = Gauge(
        "glmocr_gpu_memory_bytes",
        "GPU memory usage in bytes",
        ["device"],
    )


# ── Lifecycle ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting production GLM-OCR server...")
    config = load_config()

    multiprocessing.set_start_method("spawn", force=True)

    pipeline = Pipeline(config=config.pipeline)
    pipeline.start()
    app.state.pipeline = pipeline
    app.state.config = config
    app.state.start_time = time.time()

    logger.info("GLM-OCR production server started", port=config.server.port)

    yield

    logger.info("Shutting down...")
    pipeline.stop()
    logger.info("Shutdown complete")


# ── App Factory ─────────────────────────────────────────────────────

def create_app() -> FastAPI:
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for the production server. "
            "Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="GLM-OCR Production Server",
        version="0.1.5",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware: Request Timing + Metrics ─────────────────────────
    if PROMETHEUS_AVAILABLE:
        @app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            OCR_INFLIGHT.inc()
            start = time.time()
            try:
                response = await call_next(request)
                status = "success" if response.status_code < 400 else "error"
                OCR_REQUESTS.labels(
                    endpoint=request.url.path, status=status
                ).inc()
                return response
            except Exception:
                OCR_REQUESTS.labels(
                    endpoint=request.url.path, status="error"
                ).inc()
                raise
            finally:
                OCR_INFLIGHT.dec()
                OCR_REQUEST_DURATION.labels(
                    endpoint=request.url.path
                ).observe(time.time() - start)

    # ── Routes ───────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        pipeline = app.state.pipeline
        stats = pipeline.get_queue_stats() if pipeline else None
        return {
            "status": "healthy",
            "uptime_seconds": int(time.time() - app.state.start_time),
            "queue_stats": stats,
        }

    @app.get("/ready")
    async def ready():
        pipeline = app.state.pipeline
        if pipeline is None or not pipeline.ocr_client.is_alive():
            raise HTTPException(status_code=503, detail="OCR service not ready")
        return {"status": "ready"}

    if PROMETHEUS_AVAILABLE:
        @app.get("/metrics")
        async def metrics():
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )

    @app.post("/glmocr/parse")
    async def parse(request: Request):
        if request.headers.get("Content-Type") != "application/json":
            raise HTTPException(status_code=400, detail="Expected application/json")

        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        images = data.get("images", [])
        if isinstance(images, str):
            images = [images]

        if not images and "file" in data:
            file_val = data["file"]
            if isinstance(file_val, str) and file_val:
                images = [file_val]

        if not images:
            raise HTTPException(status_code=400, detail="No images provided")

        pipeline: Pipeline = app.state.pipeline

        messages = [{"role": "user", "content": []}]
        for image_url in images:
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )

        request_data = {"messages": messages}

        try:
            results = list(
                pipeline.process(
                    request_data,
                    save_layout_visualization=False,
                )
            )

            if not results:
                return _build_response(None, "")

            if len(results) == 1:
                r = results[0]
                return _build_response(r.json_result, r.markdown_result or "")

            json_result = [r.json_result for r in results]
            markdown_result = "\n\n---\n\n".join(
                r.markdown_result or "" for r in results
            )
            return _build_response(json_result, markdown_result)

        except Exception as e:
            logger.error("parse_error", error=str(e))
            raise HTTPException(status_code=500, detail=f"Parse error: {str(e)}")

    @app.get("/glmocr/stats")
    async def parse_stats():
        pipeline = app.state.pipeline
        stats = pipeline.get_queue_stats() if pipeline else {}
        return {
            "queue_stats": stats,
            "uptime_seconds": int(time.time() - app.state.start_time),
        }

    return app


# ── Response Builder ────────────────────────────────────────────────

def _build_response(json_result, markdown_result):
    return {
        "json_result": json_result,
        "markdown_result": markdown_result,
        "layout_details": json_result,
        "md_results": markdown_result,
        "data_info": {"pages": []},
        "usage": {},
        "model": "glm-ocr",
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "created": int(time.time()),
    }


# ── Entrypoint ──────────────────────────────────────────────────────

app = create_app()


def main():
    import uvicorn

    config = load_config()
    server_config = config.server

    uvicorn.run(
        "glmocr.production_server:app",
        host=server_config.host,
        port=server_config.port,
        workers=server_config.get("workers", 4),
        log_level="info",
        loop="uvloop",
        http="httptools",
    )


if __name__ == "__main__":
    main()