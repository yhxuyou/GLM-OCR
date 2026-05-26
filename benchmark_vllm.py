"""vLLM direct concurrency benchmark.

Bypasses the GLM-OCR queue server and hits vLLM directly with realistic
VLM payloads (base64-encoded images + OCR prompts) to diagnose GPU
utilisation issues.

Key features:
    - True async concurrency (asyncio + aiohttp), not ThreadPoolExecutor
    - Realistic image payloads (base64-encoded PNG/JPEG)
    - Gradual ramp-up mode to find the saturation point
    - Per-request latency tracking (p50 / p95 / p99)
    - Token throughput measurement
    - vLLM health / metrics polling

Usage:
    # Basic: 20 concurrent requests, 3 rounds
    python benchmark_vllm.py --url http://localhost:8080 --concurrency 20 --rounds 3

    # Ramp-up: test 1, 2, 4, 8, 16, 32, 64 concurrency
    python benchmark_vllm.py --url http://localhost:8080 --ramp-up

    # Custom image
    python benchmark_vllm.py --url http://localhost:8080 --image /path/to/doc.png

    # Text-only (no image) — for baseline comparison
    python benchmark_vllm.py --url http://localhost:8080 --no-image --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:
    print("aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


# ── Payload generation ─────────────────────────────────────────────────────


def _generate_test_image(
    width: int = 800,
    height: int = 600,
    fmt: str = "JPEG",
    quality: int = 85,
) -> str:
    """Generate a base64-encoded test image with text and table."""
    if Image is None:
        raise ImportError("Pillow required for image generation")

    img = Image.new("RGB", (width, height), (255, 255, 255))
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        draw.rectangle([50, 50, 750, 100], outline=(0, 0, 200), width=2)
        draw.text((60, 60), "Test Document - OCR Benchmark", fill=(0, 0, 200))

        draw.rectangle([50, 120, 750, 400], outline=(100, 100, 100), width=1)
        lines = [
            "This is a test document for OCR processing benchmark.",
            "",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming document processing.",
            "Deep neural networks extract text with high accuracy.",
            "",
            "Complex layouts with tables and formulas are detected.",
            "E = mc^2 is one of the most famous formulas in physics.",
            "",
            "Pipeline stages: layout detection, text recognition, formatting.",
        ]
        y = 130
        for line in lines:
            draw.text((60, y), line, fill=(50, 50, 50))
            y += 22

        draw.rectangle([50, 430, 750, 550], outline=(0, 150, 0), width=2)
        draw.line([250, 430, 250, 550], fill=(0, 150, 0), width=1)
        draw.line([500, 430, 500, 550], fill=(0, 150, 0), width=1)
        for j, h in enumerate(["Item", "Quantity", "Price"]):
            draw.text((60 + j * 250, 440), h, fill=(0, 100, 0))
        for j, row in enumerate([["Laptop", "2", "$1200"], ["Monitor", "3", "$350"]]):
            for k, cell in enumerate(row):
                draw.text((60 + k * 250, 470 + j * 25), cell, fill=(50, 50, 50))
    except Exception:
        pass

    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_image_as_base64(path: str) -> str:
    """Load an image file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_vlm_request(
    image_b64: Optional[str] = None,
    model: str = "glm-ocr",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    use_tool_call: bool = True,
) -> Dict[str, Any]:
    """Build an OpenAI-compatible VLM request payload."""
    content: List[Dict[str, Any]] = []

    if image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        })

    if use_tool_call:
        content.append({
            "type": "text",
            "text": "Text Recognition:",
        })
    else:
        content.append({
            "type": "text",
            "text": "Please recognize all text in this document image and output the content.",
        })

    messages = [{"role": "user", "content": content}]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if use_tool_call:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "add_text_block",
                    "description": "Add a text block",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "label": {"type": "string"},
                            "content": {"type": "string"},
                            "bbox_2d": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "task_type": {"type": "string"},
                        },
                        "required": ["index", "label", "content", "bbox_2d", "task_type"],
                    },
                },
            }
        ]
        payload["tool_choice"] = "auto"

    return payload


# ── Benchmark core ─────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    success: bool = False
    status_code: int = 0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str = ""


@dataclass
class BenchmarkStats:
    concurrency: int = 0
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    total_time_s: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0

    @property
    def rps(self) -> float:
        return self.successful / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def prompt_tps(self) -> float:
        return self.prompt_tokens_total / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def completion_tps(self) -> float:
        return self.completion_tokens_total / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def total_tps(self) -> float:
        return (self.prompt_tokens_total + self.completion_tokens_total) / self.total_time_s if self.total_time_s > 0 else 0

    def p50(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0

    def p95(self) -> float:
        if not self.latencies_ms:
            return 0
        s = sorted(self.latencies_ms)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    def p99(self) -> float:
        if not self.latencies_ms:
            return 0
        s = sorted(self.latencies_ms)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    def summary(self) -> str:
        return (
            f"concurrency={self.concurrency} | "
            f"req/s={self.rps:.1f} | "
            f"ok={self.successful}/{self.total_requests} | "
            f"latency p50={self.p50():.0f}ms p95={self.p95():.0f}ms p99={self.p99():.0f}ms | "
            f"prompt_tps={self.prompt_tps:.0f} completion_tps={self.completion_tps:.0f} total_tps={self.total_tps:.0f}"
        )


async def _send_one(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    request_id: int,
    timeout_s: float = 120,
) -> RequestResult:
    """Send a single request and measure latency."""
    result = RequestResult()
    t0 = time.monotonic()
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            elapsed = (time.monotonic() - t0) * 1000
            result.status_code = resp.status
            result.latency_ms = elapsed

            if resp.status == 200:
                body = await resp.json()
                usage = body.get("usage", {})
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)
                result.total_tokens = usage.get("total_tokens", 0)
                result.success = True
            else:
                text = await resp.text()
                result.error = f"HTTP {resp.status}: {text[:200]}"
    except asyncio.TimeoutError:
        result.latency_ms = (time.monotonic() - t0) * 1000
        result.error = "timeout"
    except Exception as e:
        result.latency_ms = (time.monotonic() - t0) * 1000
        result.error = str(e)[:200]
    return result


async def run_benchmark(
    url: str,
    image_b64: Optional[str],
    concurrency: int,
    total_requests: int,
    model: str,
    max_tokens: int,
    timeout_s: float = 120,
    connection_limit: int = 256,
) -> BenchmarkStats:
    """Run a single benchmark at the given concurrency level."""
    payload = build_vlm_request(
        image_b64=image_b64,
        model=model,
        max_tokens=max_tokens,
    )

    connector = aiohttp.TCPConnector(limit=connection_limit, limit_per_host=connection_limit)
    stats = BenchmarkStats(concurrency=concurrency, total_requests=total_requests)

    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)
        t_start = time.monotonic()

        async def bounded_send(req_id: int) -> RequestResult:
            async with sem:
                return await _send_one(session, url, payload, req_id, timeout_s)

        tasks = [bounded_send(i) for i in range(total_requests)]
        results: List[RequestResult] = await asyncio.gather(*tasks)

        stats.total_time_s = time.monotonic() - t_start

    for r in results:
        if r.success:
            stats.successful += 1
            stats.latencies_ms.append(r.latency_ms)
            stats.prompt_tokens_total += r.prompt_tokens
            stats.completion_tokens_total += r.completion_tokens
        else:
            stats.failed += 1
            if r.latency_ms > 0:
                stats.latencies_ms.append(r.latency_ms)

    return stats


async def check_vllm_health(url: str) -> Optional[Dict]:
    """Try to fetch vLLM health/metrics."""
    endpoints = ["/health", "/v1/models"]
    for ep in endpoints:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}{ep}", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        try:
                            return await resp.json()
                        except Exception:
                            return {"status": "ok", "endpoint": ep}
        except Exception:
            continue
    return None


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Direct Concurrency Benchmark")
    parser.add_argument("--url", default="http://localhost:8080", help="vLLM server URL")
    parser.add_argument("--concurrency", "-c", type=int, default=20, help="Concurrent requests")
    parser.add_argument("--requests", "-n", type=int, default=0, help="Total requests (0 = 3×concurrency)")
    parser.add_argument("--rounds", type=int, default=1, help="Repeat benchmark N times")
    parser.add_argument("--model", default="glm-ocr", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=4096, help="max_tokens per request")
    parser.add_argument("--timeout", type=float, default=120, help="Request timeout (seconds)")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--no-image", action="store_true", help="Send text-only requests (no image)")
    parser.add_argument("--image-width", type=int, default=800, help="Generated image width")
    parser.add_argument("--image-height", type=int, default=600, help="Generated image height")
    parser.add_argument(
        "--ramp-up",
        action="store_true",
        help="Test multiple concurrency levels: 1, 2, 4, 8, 16, 32, 64",
    )
    parser.add_argument("--connection-limit", type=int, default=256, help="aiohttp connection pool limit")
    return parser.parse_args()


async def main():
    args = parse_args()

    api_url = f"{args.url}/v1/chat/completions"

    print()
    print("=" * 70)
    print("  vLLM Direct Concurrency Benchmark")
    print("=" * 70)
    print(f"  Target:       {api_url}")
    print(f"  Model:        {args.model}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Timeout:      {args.timeout}s")
    print()

    health = await check_vllm_health(args.url)
    if health:
        print(f"  vLLM health:  {health}")
    else:
        print("  WARNING: Cannot reach vLLM server. Is it running?")
        print()
        sys.exit(1)

    # Prepare image
    image_b64: Optional[str] = None
    if not args.no_image:
        if args.image:
            print(f"  Image:        {args.image} (file)")
            image_b64 = _load_image_as_base64(args.image)
        else:
            print(f"  Image:        generated {args.image_width}x{args.image_height}")
            image_b64 = _generate_test_image(
                width=args.image_width,
                height=args.image_height,
            )
        print(f"  Image size:   {len(image_b64) / 1024:.0f} KB (base64)")
    else:
        print("  Image:        NONE (text-only)")

    print()

    # Determine concurrency levels
    if args.ramp_up:
        levels = [1, 2, 4, 8, 16, 32, 64]
    else:
        levels = [args.concurrency]

    all_stats: List[BenchmarkStats] = []

    for concurrency in levels:
        for round_idx in range(args.rounds):
            total = args.requests if args.requests > 0 else concurrency * 3
            label = f"c={concurrency}"
            if args.rounds > 1:
                label += f" round={round_idx + 1}"

            print(f"  [{label}] Sending {total} requests ...", end=" ", flush=True)

            stats = await run_benchmark(
                url=api_url,
                image_b64=image_b64,
                concurrency=concurrency,
                total_requests=total,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
                connection_limit=args.connection_limit,
            )
            all_stats.append(stats)

            print(stats.summary())

            if stats.failed > 0 and stats.failed == stats.total_requests:
                print(f"    ALL REQUESTS FAILED — stopping ramp-up")
                break

            if concurrency <= 4:
                await asyncio.sleep(1)

        else:
            continue
        break

    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Conc':>5} | {'req/s':>8} | {'p50ms':>7} | {'p95ms':>7} | {'p99ms':>7} | {'prompt_tps':>11} | {'comp_tps':>9} | {'total_tps':>10} | {'ok/total':>9}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*11}-+-{'-'*9}-+-{'-'*10}-+-{'-'*9}")
    for s in all_stats:
        print(
            f"  {s.concurrency:>5} | {s.rps:>8.1f} | {s.p50():>7.0f} | {s.p95():>7.0f} | {s.p99():>7.0f} | "
            f"{s.prompt_tps:>11.0f} | {s.completion_tps:>9.0f} | {s.total_tps:>10.0f} | "
            f"{s.successful}/{s.total_requests}"
        )
    print()

    # Diagnosis
    if all_stats:
        best = max(all_stats, key=lambda s: s.rps)
        print(f"  Peak throughput: {best.rps:.1f} req/s at concurrency={best.concurrency}")
        print(f"  Peak token throughput: {best.total_tps:.0f} tokens/s")

        if best.prompt_tps < 5000:
            print()
            print("  DIAGNOSIS: prompt_tps is very low (< 5000). Likely causes:")
            print("    1. --max-num-batched-tokens too small (try 49152 or 65536)")
            print("    2. --gpu-memory-utilization too low (try 0.95)")
            print("    3. --max-model-len too large (try 32768)")
            print("    4. Image not being sent (check --no-image flag)")

        if best.rps < 2 and best.p50() > 10000:
            print()
            print("  DIAGNOSIS: Very high latency (>10s) with low throughput. Likely:")
            print("    1. Requests are being serialized (check vLLM --max-num-seqs)")
            print("    2. KV cache too small, causing frequent eviction")
            print("    3. Model is too large for single GPU (consider --tensor-parallel-size)")

    print()


if __name__ == "__main__":
    asyncio.run(main())