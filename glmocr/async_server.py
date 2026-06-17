"""GLM-OCR Async FastAPI Server

Async document processing server with real-time progress updates via WebSocket.

Endpoints:
  - POST /parse/async: Submit document for async processing
  - GET /parse/status/{doc_id}: Query processing progress
  - GET /parse/result/{doc_id}: Get final result
  - WebSocket /ws/{doc_id}: Real-time progress updates
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
import uuid
from typing import Any, Dict, Optional

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
except ImportError as e:  # pragma: no cover
    FastAPI = None  # type: ignore
    _FASTAPI_IMPORT_ERROR = e

from glmocr.aggregator import RegionAggregator
from glmocr.async_pipeline import AsyncPipeline
from glmocr.config import GlmOcrConfig, load_config
from glmocr.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(config: GlmOcrConfig) -> FastAPI:
    """Create a FastAPI application for async document processing.

    Args:
        config: GlmOcrConfig instance containing server and pipeline settings.

    Returns:
        FastAPI application instance.
    """
    if FastAPI is None:
        raise ImportError(
            "FastAPI server support requires the optional server extra. "
            "Install with: pip install 'glmocr[server]'"
        ) from _FASTAPI_IMPORT_ERROR

    app = FastAPI(
        title="GLM-OCR Async Server",
        description="Async document processing with real-time progress updates",
        version="1.0.0",
    )

    # Store config in app.state for access in lifecycle events
    app.state.config = config

    @app.on_event("startup")
    async def startup_event():
        """Initialize pipeline and aggregator on application startup."""
        logger.info("Starting GLM-OCR Async Server...")

        # Initialize AsyncPipeline
        pipeline = AsyncPipeline(config=config.pipeline)
        await pipeline.start()
        app.state.pipeline = pipeline
        logger.info("AsyncPipeline initialized")

        # Initialize RegionAggregator
        redis_config = config.pipeline.redis
        aggregator = RegionAggregator(
            redis_url=redis_config.url,
            key_prefix=redis_config.key_prefix,
            max_connections=redis_config.max_connections,
        )
        await aggregator.connect()
        app.state.aggregator = aggregator
        logger.info("RegionAggregator initialized")

        logger.info("GLM-OCR Async Server started successfully")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup connections on application shutdown."""
        logger.info("Shutting down GLM-OCR Async Server...")

        # Stop pipeline
        if hasattr(app.state, "pipeline"):
            await app.state.pipeline.stop()
            logger.info("AsyncPipeline stopped")

        # Disconnect aggregator
        if hasattr(app.state, "aggregator"):
            await app.state.aggregator.disconnect()
            logger.info("RegionAggregator disconnected")

        logger.info("GLM-OCR Async Server shutdown complete")

    @app.post("/parse/async")
    async def parse_async(file: UploadFile = File(...)) -> Dict[str, Any]:
        """Submit a document for async processing.

        Accepts multipart/form-data with a file upload. Generates a unique
        document ID and starts background processing.

        Args:
            file: Uploaded file (image or PDF).

        Returns:
            Dictionary with doc_id and status.

        Raises:
            HTTPException: If file upload fails or processing cannot start.
        """
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Generate unique document ID
        doc_id = str(uuid.uuid4())

        try:
            # Read file content
            file_content = await file.read()
            if not file_content:
                raise HTTPException(status_code=400, detail="Empty file uploaded")

            # Get pipeline and aggregator from app.state
            pipeline: Pipeline = app.state.pipeline
            aggregator: RegionAggregator = app.state.aggregator

            # NOTE: Do NOT register_document here with total_regions=1.
            # The AsyncPipeline.process_async will register with the accurate
            # region count after layout detection, avoiding wasteful 0/1 polling.

            # Start background processing task
            asyncio.create_task(
                _process_document_background(
                    doc_id=doc_id,
                    file_content=file_content,
                    filename=file.filename,
                    pipeline=pipeline,
                    aggregator=aggregator,
                )
            )

            return {"doc_id": doc_id, "status": "processing"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to submit document for processing: %s", e)
            logger.debug(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to submit document: {str(e)}",
            )

    @app.get("/parse/status/{doc_id}")
    async def get_status(doc_id: str) -> Dict[str, Any]:
        """Query processing progress for a document.

        Args:
            doc_id: Document identifier returned by /parse/async.

        Returns:
            Dictionary with completed count, total count, and status.

        Raises:
            HTTPException: If document not found.
        """
        aggregator: RegionAggregator = app.state.aggregator

        try:
            progress = await aggregator.get_progress(doc_id)

            # Map aggregator status to API status
            status = progress["status"]
            if status == "unknown":
                return {
                    "completed": 0,
                    "total": 0,
                    "status": "not_found",
                }
            elif status == "complete":
                return {
                    "completed": progress["completed"],
                    "total": progress["total"],
                    "status": "completed",
                }
            else:
                return {
                    "completed": progress["completed"],
                    "total": progress["total"],
                    "status": "processing",
                }

        except Exception as e:
            logger.error("Failed to get status for doc %s: %s", doc_id, e)
            logger.debug(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get status: {str(e)}",
            )

    @app.get("/parse/result/{doc_id}")
    async def get_result(doc_id: str) -> Dict[str, Any]:
        """Get the final OCR result for a completed document.

        Args:
            doc_id: Document identifier.

        Returns:
            Aggregated OCR result dictionary.

        Raises:
            HTTPException: 404 if not found, 202 if still processing.
        """
        aggregator: RegionAggregator = app.state.aggregator

        try:
            # Check if document exists
            progress = await aggregator.get_progress(doc_id)
            if progress["status"] == "unknown":
                raise HTTPException(status_code=404, detail="Document not found")

            # Check if complete
            is_complete = await aggregator.is_complete(doc_id)
            if not is_complete:
                raise HTTPException(
                    status_code=202,
                    detail="Document processing not yet complete",
                )

            # Get result
            result = await aggregator.get_result(doc_id)
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to get result for doc %s: %s", doc_id, e)
            logger.debug(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get result: {str(e)}",
            )

    @app.websocket("/ws/{doc_id}")
    async def websocket_endpoint(websocket: WebSocket, doc_id: str):
        """WebSocket endpoint for real-time progress updates.

        Connects to a document processing task and sends progress updates
        every 1-2 seconds until completion.

        Args:
            websocket: WebSocket connection.
            doc_id: Document identifier to monitor.
        """
        await websocket.accept()

        aggregator: RegionAggregator = app.state.aggregator

        try:
            # Wait for document to be registered (layout detection may take a moment)
            # Retry for up to 30 seconds before giving up
            for _ in range(60):
                progress = await aggregator.get_progress(doc_id)
                if progress["status"] != "unknown":
                    break
                await asyncio.sleep(0.5)
            else:
                await websocket.send_json(
                    {"type": "error", "message": "Document not found (registration timeout)"}
                )
                await websocket.close(code=1008, reason="Document not found")
                return

            # Poll for progress updates
            while True:
                progress = await aggregator.get_progress(doc_id)

                # Send progress update
                await websocket.send_json(
                    {
                        "type": "progress",
                        "completed": progress["completed"],
                        "total": progress["total"],
                    }
                )

                # Check if complete
                if progress["status"] == "complete":
                    # Get final result
                    result = await aggregator.get_result(doc_id)
                    await websocket.send_json(
                        {"type": "complete", "result": result}
                    )
                    break

                # Wait before next poll
                await asyncio.sleep(1.5)

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected from doc %s", doc_id)
        except Exception as e:
            logger.error("WebSocket error for doc %s: %s", doc_id, e)
            logger.debug(traceback.format_exc())
            try:
                await websocket.send_json(
                    {"type": "error", "message": str(e)}
                )
                await websocket.close(code=1011, reason="Internal error")
            except Exception:
                pass

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    return app


async def _process_document_background(
    doc_id: str,
    file_content: bytes,
    filename: str,
    pipeline: AsyncPipeline,
    aggregator: RegionAggregator,
) -> None:
    """Background task to process a document using AsyncPipeline.

    Args:
        doc_id: Document identifier.
        file_content: Raw file content (bytes).
        filename: Original filename.
        pipeline: AsyncPipeline instance for processing.
        aggregator: RegionAggregator for tracking progress.
    """
    try:
        logger.info("Starting background processing for doc %s", doc_id)

        # Determine file type from filename or content
        file_ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        # Build request data for pipeline
        # For now, we'll save the file temporarily and pass as file:// URL
        # In a production system, you might want to handle this differently
        import tempfile
        import os

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file_ext}" if file_ext else ""
        ) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            # Build pipeline request
            file_url = f"file://{tmp_path}"
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": file_url}}],
                }
            ]
            request_data = {"messages": messages}

            # Use AsyncPipeline's process_async method
            # This will submit regions to vLLM asynchronously and return immediately
            await pipeline.process_async(request_data, doc_id, aggregator)

            logger.info("Background processing submitted for doc %s", doc_id)

        finally:
            # Cleanup temporary file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        logger.error("Background processing failed for doc %s: %s", doc_id, e)
        logger.debug(traceback.format_exc())
        # Register and mark as failed so callers don't see "not_found"
        try:
            await aggregator.register_document(doc_id, total_regions=1)
            await aggregator.on_region_complete(
                doc_id,
                "error",
                {"error": str(e), "json_result": None, "markdown_result": ""},
            )
        except Exception:
            logger.error("Failed to register error state for doc %s", doc_id)


def main():
    """Main entrypoint for the async server."""
    parser = argparse.ArgumentParser(description="GLM-OCR Async Server")
    parser.add_argument(
        "--config", type=str, default=None, help="Config file path"
    )
    parser.add_argument(
        "--host", type=str, default=None, help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to bind to"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of Uvicorn worker processes"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Override host/port if provided
        if args.host:
            config.server.host = args.host
        if args.port:
            config.server.port = args.port

        # Configure logging
        log_level = args.log_level or config.logging.level
        configure_logging(level=log_level)

        # Create app
        app = create_app(config)

        # Import uvicorn here to avoid requiring it for non-server usage
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "Uvicorn is required to run the async server. "
                "Install with: pip install 'glmocr[server]'"
            ) from e

        # Start server
        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "GLM-OCR Async Server starting on %s:%d...",
            config.server.host,
            config.server.port,
        )
        logger.info("API endpoints:")
        logger.info("  POST /parse/async - Submit document for async processing")
        logger.info("  GET  /parse/status/{doc_id} - Query processing progress")
        logger.info("  GET  /parse/result/{doc_id} - Get final result")
        logger.info("  WS   /ws/{doc_id} - Real-time progress updates")
        logger.info("=" * 60)
        logger.info("")

        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level=log_level.lower(),
            workers=args.workers,
        )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Error: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
