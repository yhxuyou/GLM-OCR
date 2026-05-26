"""Mock vLLM server for GLM-OCR testing."""

from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import threading
import time
import uuid

try:
    from flask import Flask, request, jsonify
except ImportError:
    print("Error: Flask required. pip install flask")
    sys.exit(1)

app = Flask(__name__)

CONFIG = {
    "base_delay": 1.5,
    "delay_jitter": 1.0,
    "fail_rate": 0.0,
    "max_concurrent": 10,
}
SEMAPHORE: threading.BoundedSemaphore | None = None

STATS = {"total_requests": 0, "successful": 0, "failed": 0, "rejected": 0, "active_requests": 0}
STATS_LOCK = threading.Lock()

_CONTENT_TYPES = [
    "人工智能是计算机科学的重要分支。",
    "深度学习在图像识别领域取得了突破性进展。",
    "大语言模型广泛应用于自然语言处理任务。",
]


def _generate_mock_response() -> str:
    tools = [
        {"name": "add_text_block", "arguments": json.dumps({"index": 0, "label": "text", "content": random.choice(_CONTENT_TYPES), "bbox_2d": [100, 100, 500, 23]}, ensure_ascii=False)},
        {"name": "add_text_block", "arguments": json.dumps({"index": 0, "label": "text", "content": "表格内容：姓名|年龄|城市", "bbox_2d": [100, 200, 500, 80], "is_table": True}, ensure_ascii=False)},
        {"name": "add_text_block", "arguments": json.dumps({"index": 0, "label": "text", "content": "公式: E = mc^2", "bbox_2d": [100, 300, 500, 30], "is_formula": True}, ensure_ascii=False)},
    ]
    return json.dumps(random.sample(tools, k=random.randint(1, 3)), ensure_ascii=False)


@app.route("/health", methods=["GET"])
def health():
    with STATS_LOCK:
        s = dict(STATS)
    return jsonify({"status": "ok", "stats": s}), 200


@app.route("/stats", methods=["GET"])
def stats():
    with STATS_LOCK:
        return jsonify(dict(STATS)), 200


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/api/generate", methods=["POST"])
def chat_completions():
    sem = SEMAPHORE
    if sem is not None and not sem.acquire(blocking=False):
        with STATS_LOCK:
            STATS["total_requests"] += 1
            STATS["rejected"] += 1
        return jsonify({"error": {"message": "Server overloaded", "code": "overloaded"}}), 503

    with STATS_LOCK:
        STATS["total_requests"] += 1
        STATS["active_requests"] += 1

    try:
        if random.random() < CONFIG["fail_rate"]:
            with STATS_LOCK:
                STATS["failed"] += 1
            return jsonify({"error": {"message": "Internal error"}}), 500

        delay = CONFIG["base_delay"] + random.uniform(-CONFIG["delay_jitter"], CONFIG["delay_jitter"])
        delay = max(0.1, delay)
        time.sleep(delay)

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "glm-ocr",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": _generate_mock_response()}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1024, "completion_tokens": 100, "total_tokens": 1124},
        }
        with STATS_LOCK:
            STATS["successful"] += 1
        return jsonify(response), 200
    except Exception as e:
        with STATS_LOCK:
            STATS["failed"] += 1
        return jsonify({"error": {"message": str(e)}}), 500
    finally:
        if sem is not None:
            sem.release()
        with STATS_LOCK:
            STATS["active_requests"] = max(0, STATS["active_requests"] - 1)


def main():
    global SEMAPHORE, CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--jitter", type=float, default=1.0)
    parser.add_argument("--fail-rate", type=float, default=0.0)
    parser.add_argument("--max-concurrent", type=int, default=10)
    args = parser.parse_args()

    CONFIG.update({"base_delay": args.delay, "delay_jitter": args.jitter, "fail_rate": args.fail_rate, "max_concurrent": args.max_concurrent})
    SEMAPHORE = threading.BoundedSemaphore(args.max_concurrent)

    print(f"Mock vLLM running on {args.host}:{args.port} (delay={args.delay}s±{args.jitter}s, max_concurrent={args.max_concurrent})")
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
