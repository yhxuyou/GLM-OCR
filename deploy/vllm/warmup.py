#!/usr/bin/env python3
"""
vLLM GLM-OCR 模型预热脚本

在服务启动后立即执行一次推理，避免首次请求的超时。
支持多轮预热以触发 JIT 编译和显存分配优化。
"""
import argparse
import json
import time
import base64
import io
import urllib.request
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


# ── 生成测试图像 ──────────────────────────────────────────────────────

def create_dummy_image(width: int = 512, height: int = 512) -> str:
    """创建一张纯色测试图片，返回 base64 data URI."""
    if Image is None:
        raise ImportError("Pillow is required. Install: pip install Pillow")

    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ── 发送请求 ──────────────────────────────────────────────────────────

def send_warmup_request(
    server_url: str,
    model_name: str,
    image_data_uri: str,
    timeout: int = 60,
) -> dict:
    """向 vLLM 发送一次预热请求."""
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ],
        "max_tokens": 128,
        "temperature": 0.0,
    }

    req = urllib.request.Request(
        url=f"{server_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - start

    return {"elapsed_seconds": round(elapsed, 2), "response": result}


# ── 健康检查 ──────────────────────────────────────────────────────────

def wait_for_server(
    server_url: str,
    max_retries: int = 30,
    interval: float = 5.0,
) -> bool:
    """等待 vLLM 服务就绪."""
    health_url = f"{server_url}/health"
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[OK] 服务就绪 (尝试 {attempt}/{max_retries})")
                    return True
        except Exception:
            pass
        print(f"[..] 等待服务启动... (尝试 {attempt}/{max_retries}, 每 {interval}s)")
        time.sleep(interval)
    return False


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="vLLM GLM-OCR 模型预热")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="vLLM 服务地址 (默认: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="glm-ocr",
        help="模型名称 (默认: glm-ocr)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="预热轮数 (默认: 3, 更多轮次让 JIT 充分优化)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="测试图像大小 (默认: 512, 越小越快)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="启动后等待服务就绪再预热",
    )
    args = parser.parse_args()

    server_url = args.url.rstrip("/")
    print(f"vLLM GLM-OCR 模型预热脚本")
    print(f"  服务地址: {server_url}")
    print(f"  模型名称: {args.model}")
    print(f"  预热轮数: {args.rounds}")
    print(f"  图像大小: {args.image_size}x{args.image_size}")
    print()

    # 1. 等待服务就绪
    if args.wait:
        print("[1/3] 等待 vLLM 服务就绪...")
        if not wait_for_server(server_url):
            print("[FAIL] 服务未能就绪，退出")
            exit(1)
        print()

    # 2. 创建测试图像
    print("[2/3] 生成测试图像...")
    image_uri = create_dummy_image(args.image_size, args.image_size)
    image_size_kb = len(image_uri) * 3 / 4 / 1024
    print(f"  测试图像大小: {image_size_kb:.0f} KB (base64)")
    print()

    # 3. 执行预热
    print(f"[3/3] 执行 {args.rounds} 轮预热推理...")
    print(f"  {'#':>3}  {'耗时(s)':>8}  {'状态':>10}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*10}")

    timings = []
    for i in range(1, args.rounds + 1):
        try:
            result = send_warmup_request(server_url, args.model, image_uri)
            elapsed = result["elapsed_seconds"]
            timings.append(elapsed)
            status = "OK" if result["response"].get("choices") else "ERR"
            print(f"  {i:>3}  {elapsed:>8.2f}  {status:>10}")
        except Exception as e:
            print(f"  {i:>3}  {'FAIL':>8}  {str(e)[:30]:>10}")

    print()
    if timings:
        print(f"  预热完成! 耗时统计:")
        print(f"    首次: {timings[0]:.2f}s (最慢, 含显存分配)")
        if len(timings) > 1:
            print(f"    末次: {timings[-1]:.2f}s (最快, JIT 编译完成)")
            print(f"    平均: {sum(timings)/len(timings):.2f}s")

        if len(timings) > 1 and timings[-1] < timings[0] * 0.5:
            print(f"  性能提升: {((1 - timings[-1]/timings[0]) * 100):.0f}% (JIT 优化效果)")
    else:
        print("  所有预热请求均失败!")

    print()
    print("服务已就绪，可以处理业务请求。")


if __name__ == "__main__":
    main()