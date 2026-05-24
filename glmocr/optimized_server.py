"""GLM-OCR SDK Flask service with optimized pipeline."""

import os
import sys
import time
import traceback
import uuid
import multiprocessing
from typing import TYPE_CHECKING

try:
    from flask import Flask, request, jsonify

    _FLASK_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover
    Flask = None  # type: ignore
    request = None  # type: ignore
    jsonify = None  # type: ignore
    _FLASK_IMPORT_ERROR = e

from glmocr.pipeline import OptimizedPipeline
from glmocr.layout import OptimizedPPDocLayoutDetector
from glmocr.config import load_config
from glmocr.utils.logging import get_logger, configure_logging

if TYPE_CHECKING:
    from glmocr.config import GlmOcrConfig

logger = get_logger(__name__)

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def _build_response(json_result, markdown_result):
    """Build response dict with both SDK native and MaaS-compatible fields."""
    return {
        "json_result": json_result,
        "markdown_result": markdown_result,
        "layout_details": json_result,
        "md_results": markdown_result,
        "data_info": {"pages": []},
        "usage": {},
        "model": "glm-ocr-optimized",
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "created": int(time.time()),
    }


def create_app(config: "GlmOcrConfig") -> Flask:
    """Create a Flask app with optimized pipeline.

    Args:
        config: GlmOcrConfig instance.

    Returns:
        Flask app instance.
    """
    if Flask is None:
        raise ImportError(
            "Flask server support requires the optional server extra. "
            "Install with: pip install 'glmocr[server]'"
        ) from _FLASK_IMPORT_ERROR

    app = Flask(__name__)

    layout_detector = OptimizedPPDocLayoutDetector(config.pipeline.layout)
    pipeline = OptimizedPipeline(config=config.pipeline, layout_detector=layout_detector)

    app.config["pipeline"] = pipeline
    app.config["doc_config"] = config
    app.config["optimized"] = True

    @app.route("/glmocr/parse", methods=["POST"])
    def parse():
        """Document parsing endpoint with optimized processing.

        Request:
            {
                "images": ["url1", "url2", ...],  # image URLs (http/https/file/data)
                "preprocess_options": {...},      # Optional preprocessing options
                "postprocess_options": {...},     # Optional postprocessing options
            }

        Response:
            {
                "json_result": {...},
                "markdown_result": "...",
                "optimized": true
            }
        """
        if request.headers.get("Content-Type") != "application/json":
            return (
                jsonify(
                    {"error": "Invalid Content-Type. Expected 'application/json'."}
                ),
                400,
            )

        try:
            data = request.json
        except Exception:
            return jsonify({"error": "Invalid JSON payload"}), 400

        images = data.get("images", [])
        if isinstance(images, str):
            images = [images]

        if not images and "file" in data:
            file_val = data["file"]
            if isinstance(file_val, str) and file_val:
                images = [file_val]

        if not images:
            return jsonify({"error": "No images provided"}), 400

        preprocess_context = data.get("preprocess_options", {})
        postprocess_context = data.get("postprocess_options", {})

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
                    preprocess_context=preprocess_context,
                    postprocess_context=postprocess_context,
                )
            )
            if not results:
                return jsonify(_build_response(None, "")), 200
            if len(results) == 1:
                r = results[0]
                return (
                    jsonify(_build_response(r.json_result, r.markdown_result or "")),
                    200,
                )
            json_result = [r.json_result for r in results]
            markdown_result = "\n\n---\n\n".join(
                r.markdown_result or "" for r in results
            )
            return jsonify(_build_response(json_result, markdown_result)), 200

        except Exception as e:
            logger.error("Parse error: %s", e)
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok", "optimized": True}), 200

    @app.route("/glmocr/parse/enhanced", methods=["POST"])
    def parse_enhanced():
        """Enhanced document parsing endpoint with custom hooks support.

        Request:
            {
                "images": ["url1", "url2", ...],
                "enhancements": {
                    "denoise": true,
                    "contrast": true,
                    "region_filtering": true,
                    "polygon_smoothing": true
                }
            }
        """
        if request.headers.get("Content-Type") != "application/json":
            return (
                jsonify(
                    {"error": "Invalid Content-Type. Expected 'application/json'."}
                ),
                400,
            )

        try:
            data = request.json
        except Exception:
            return jsonify({"error": "Invalid JSON payload"}), 400

        images = data.get("images", [])
        if isinstance(images, str):
            images = [images]

        if not images and "file" in data:
            file_val = data["file"]
            if isinstance(file_val, str) and file_val:
                images = [file_val]

        if not images:
            return jsonify({"error": "No images provided"}), 400

        enhancements = data.get("enhancements", {})

        messages = [{"role": "user", "content": []}]
        for image_url in images:
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )

        request_data = {"messages": messages}

        postprocess_context = {"enhancements": enhancements}

        try:
            results = list(
                pipeline.process(
                    request_data,
                    save_layout_visualization=False,
                    postprocess_context=postprocess_context,
                )
            )
            if not results:
                return jsonify(_build_response(None, "")), 200
            if len(results) == 1:
                r = results[0]
                response = _build_response(r.json_result, r.markdown_result or "")
                response["enhancements_applied"] = enhancements
                return jsonify(response), 200
            json_result = [r.json_result for r in results]
            markdown_result = "\n\n---\n\n".join(
                r.markdown_result or "" for r in results
            )
            response = _build_response(json_result, markdown_result)
            response["enhancements_applied"] = enhancements
            return jsonify(response), 200

        except Exception as e:
            logger.error("Enhanced parse error: %s", e)
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

    return app


def main():
    """Main entrypoint for optimized server."""
    import argparse

    parser = argparse.ArgumentParser(description="GlmOcr Optimized Server")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)

    app = None

    try:
        config = load_config(args.config)

        log_level = args.log_level or config.logging.level
        configure_logging(level=log_level)

        app = create_app(config)

        pipeline = app.config["pipeline"]
        pipeline.start()

        server_config = config.server
        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "GlmOcr Optimized Server starting on %s:%d...",
            server_config.host,
            server_config.port,
        )
        logger.info("API endpoint: /glmocr/parse")
        logger.info("Enhanced endpoint: /glmocr/parse/enhanced")
        logger.info("=" * 60)
        logger.info("")

        app.run(
            debug=server_config.debug,
            host=server_config.host,
            port=server_config.port,
        )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Error: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        if app is not None and "pipeline" in app.config:
            app.config["pipeline"].stop()


if __name__ == "__main__":
    main()