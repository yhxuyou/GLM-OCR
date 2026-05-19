"""Medical OCR Flask Server.

Specialized OCR server for medical documents, built on top of glmocr.
"""

import os
import sys
import time
import traceback
import uuid
import multiprocessing
from typing import Optional, Dict, Any

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    request = None
    jsonify = None

from glmocr.config import load_config, GlmOcrConfig
from glmocr.utils.logging import get_logger, configure_logging

from medical_ocr.pipeline import MedicalOcrPipeline
from medical_ocr.layout_detector import MedicalLayoutDetector

logger = get_logger(__name__)

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def _build_response(
    json_result: Optional[str],
    markdown_result: Optional[str],
    extra_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build API response with medical OCR specific metadata."""
    response = {
        # SDK native fields
        "json_result": json_result,
        "markdown_result": markdown_result,
        # MaaS-compatible fields
        "layout_details": json_result,
        "md_results": markdown_result,
        "data_info": {"pages": []},
        "usage": {},
        "model": "medical-ocr",
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "created": int(time.time()),
        # Medical OCR specific
        "is_medical_ocr": True,
        "version": "0.1.0",
    }
    
    if extra_info:
        response.update(extra_info)
        
    return response


def create_app(config: GlmOcrConfig) -> Flask:
    """Create Medical OCR Flask application.

    Args:
        config: Configuration object

    Returns:
        Flask application instance
    """
    if not FLASK_AVAILABLE:
        raise ImportError(
            "Flask server support requires the optional server extra. "
            "Install with: pip install flask"
        )

    app = Flask(__name__)

    # Initialize medical-specific layout detector
    medical_layout_detector = MedicalLayoutDetector(config.pipeline.layout)
    
    # Initialize medical OCR pipeline
    pipeline = MedicalOcrPipeline(
        config=config.pipeline,
        layout_detector=medical_layout_detector
    )

    # Store in app config
    app.config["pipeline"] = pipeline
    app.config["doc_config"] = config
    app.config["is_medical_ocr"] = True

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "ok",
            "service": "medical-ocr",
            "version": "0.1.0",
            "is_medical_ocr": True
        })

    @app.route("/medical-ocr/parse", methods=["POST"])
    def parse_medical_document():
        """Medical document parsing endpoint.

        Request:
            {
                "images": ["url1", "url2", ...],  # Image URLs
                "preprocess_options": {           # Optional preprocessing
                    "enable_enhancement": true,
                    "denoise_strength": 10
                },
                "postprocess_options": {          # Optional postprocessing
                    "normalize_terminology": true,
                    "extract_medical_entities": true
                }
            }

        Response:
            Standard OCR response with medical-specific enhancements
        """
        # Validate content type
        if request.headers.get("Content-Type") != "application/json":
            return jsonify({
                "error": "Invalid Content-Type. Expected 'application/json'."
            }), 400

        # Parse request data
        try:
            data = request.json
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Get images from request
        images = data.get("images", [])
        if isinstance(images, str):
            images = [images]

        # Compatibility: support "file" field
        if not images and "file" in data:
            file_val = data["file"]
            if isinstance(file_val, str) and file_val:
                images = [file_val]

        if not images:
            return jsonify({"error": "No images provided"}), 400

        # Get processing options
        preprocess_context = data.get("preprocess_options", {})
        postprocess_context = data.get("postprocess_options", {})

        # Build request data for pipeline
        messages = [{"role": "user", "content": []}]
        for image_url in images:
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )
        request_data = {"messages": messages}

        try:
            # Process with medical pipeline
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
                result = results[0]
                return jsonify(_build_response(
                    result.json_result,
                    result.markdown_result
                )), 200

            # Multiple results
            json_result = [r.json_result for r in results]
            markdown_result = "\n\n---\n\n".join(
                r.markdown_result or "" for r in results
            )
            return jsonify(_build_response(json_result, markdown_result)), 200

        except Exception as e:
            logger.error(f"Parse error: {e}")
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

    @app.route("/medical-ocr/parse/enhanced", methods=["POST"])
    def parse_enhanced_medical():
        """Enhanced medical document parsing with advanced options.

        Request:
            {
                "images": ["url1", ...],
                "medical_enhancements": {
                    "enable_enhancement": true,
                    "normalize_units": true,
                    "extract_patient_info": true,
                    "highlight_critical_values": false
                }
            }
        """
        if request.headers.get("Content-Type") != "application/json":
            return jsonify({
                "error": "Invalid Content-Type. Expected 'application/json'."
            }), 400

        try:
            data = request.json
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
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

        medical_enhancements = data.get("medical_enhancements", {})
        postprocess_context = {
            "enhancements": medical_enhancements
        }

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
                    postprocess_context=postprocess_context,
                )
            )

            if not results:
                return jsonify(_build_response(None, "")), 200

            extra_info = {
                "medical_enhancements_applied": medical_enhancements
            }

            if len(results) == 1:
                result = results[0]
                return jsonify(_build_response(
                    result.json_result,
                    result.markdown_result,
                    extra_info
                )), 200

            json_result = [r.json_result for r in results]
            markdown_result = "\n\n---\n\n".join(
                r.markdown_result or "" for r in results
            )
            return jsonify(_build_response(
                json_result, markdown_result, extra_info
            )), 200

        except Exception as e:
            logger.error(f"Enhanced parse error: {e}")
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

    return app


def main():
    """Main entry point for Medical OCR server."""
    import argparse

    parser = argparse.ArgumentParser(description="Medical OCR Server")
    parser.add_argument(
        "--config", type=str, default=None, help="Config file path"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to listen on"
    )
    args = parser.parse_args()

    # Set multiprocessing start method
    multiprocessing.set_start_method("spawn", force=True)

    app = None

    try:
        config = load_config(args.config)

        # Configure logging
        log_level = args.log_level or getattr(config.logging, "level", "INFO")
        configure_logging(level=log_level)

        # Create app
        app = create_app(config)

        # Start pipeline
        pipeline = app.config["pipeline"]
        pipeline.start()

        # Get server config
        server_config = getattr(config, "server", None)
        host = args.host or (server_config.host if server_config else "0.0.0.0")
        port = args.port or (server_config.port if server_config else 8080)
        debug = getattr(server_config, "debug", False) if server_config else False

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Medical OCR Server starting on {host}:{port}")
        logger.info("Health check: /health")
        logger.info("API endpoint: /medical-ocr/parse")
        logger.info("Enhanced API: /medical-ocr/parse/enhanced")
        logger.info("=" * 60)
        logger.info("")

        app.run(host=host, port=port, debug=debug)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        if app is not None and "pipeline" in app.config:
            try:
                app.config["pipeline"].stop()
            except Exception as e:
                logger.warning(f"Error stopping pipeline: {e}")


if __name__ == "__main__":
    main()
