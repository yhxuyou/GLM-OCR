import json
import time
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import httpx
import base64
from app.utils.config import settings
from app.utils.logger import logger
from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, RateLimiter


class ServiceRequestError(Exception):
    """服务请求错误"""
    pass


class ServiceResponseError(Exception):
    """服务响应错误"""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


class FallbackResult:
    """Represents a degraded/fallback OCR result when the primary service is unavailable."""

    def __init__(self, page_num: int, reason: str):
        self.page_num = page_num
        self.reason = reason

    def to_block(self) -> Dict[str, Any]:
        return {
            "index": 0,
            "label": "text",
            "bbox_2d": [0, 0, 0, 0],
            "content": f"[OCR service unavailable: {self.reason}]",
            "page_num": self.page_num,
        }


class LayoutAndOCRClient:
    """Layout分析和OCR识别服务的客户端。

    生产级版本，支持：
    - 熔断器（Circuit Breaker）
    - 限流器（Rate Limiter）
    - 优雅降级（Graceful Degradation）
    - 超时控制
    - 重试与退避
    """

    DEFAULT_PROMPT = "<|PIPELINE_DOCUMENT_RECOGNITION|>"

    def __init__(
        self,
        service_url: Optional[str] = None,
        timeout: float = 120.0,
        enable_circuit_breaker: bool = True,
        enable_rate_limiter: bool = True,
        max_retries: int = 2,
        fallback_enabled: bool = True,
    ):
        self.service_url = service_url or getattr(settings, "layout_ocr_url", None)
        if not self.service_url:
            self.service_url = "http://localhost:5002/glmocr/parse"

        self.timeout = timeout
        self.max_retries = max_retries
        self.fallback_enabled = fallback_enabled
        self.async_headers = {"Content-Type": "application/json"}

        self._circuit_breaker = CircuitBreaker(
            name="ocr-service",
            fail_max=5,
            reset_timeout=60.0,
            half_open_max_calls=1,
        ) if enable_circuit_breaker else None

        self._rate_limiter = RateLimiter(
            max_rate=50.0,
            time_window=1.0,
        ) if enable_rate_limiter else None

        self._last_fallback_time = 0.0
        self._fallback_count = 0

    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        ext = Path(image_path).suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(ext, "image/png")

        return f"data:{mime_type};base64,{image_data}"

    def _create_multi_image_payload(self, image_paths: List[str]) -> Dict[str, Any]:
        if not image_paths:
            raise ValueError("至少需要一张图片")

        image_base64_list = []
        for image_path in image_paths:
            image_base64 = self._encode_image_to_base64(image_path)
            image_base64_list.append(image_base64)

        return {"images": image_base64_list}

    def _create_single_image_payload(self, image_path: str) -> Dict[str, Any]:
        return self._create_multi_image_payload([image_path])

    def _should_fallback(self) -> bool:
        """Check if we should use fallback mode based on recent failures."""
        if not self.fallback_enabled:
            return False
        if self._circuit_breaker and self._circuit_breaker.state.value == "open":
            logger.warning("circuit_breaker_open_entering_fallback")
            return True
        return False

    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        if self._circuit_breaker:
            return self._circuit_breaker.stats()
        return {"state": "disabled"}

    async def _make_request_with_retry(
        self, client: httpx.AsyncClient, url: str, payload: Dict
    ) -> List[List[Dict[str, Any]]]:
        """Make the HTTP request with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, json=payload)

                if response.status_code == 200:
                    result = response.json()
                    return self._parse_response(result)

                if response.status_code in (429, 503, 502, 504):
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else min(2 ** attempt * 0.5, 30)
                    logger.warning(
                        "ocr_service_throttled",
                        status=response.status_code,
                        attempt=attempt + 1,
                        wait_seconds=wait,
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(wait)
                        continue

                error_msg = f"Service returned status code {response.status_code}"
                try:
                    error_detail = response.json()
                    if isinstance(error_detail, dict):
                        error_detail = error_detail.get("error", {}).get("message", "")
                    if error_detail:
                        error_msg += f": {error_detail}"
                except Exception:
                    pass
                raise ServiceResponseError(response.status_code, error_msg)

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait = min(2 ** attempt * 1.0, 30)
                    logger.warning(
                        "ocr_service_connection_error",
                        error=str(e),
                        attempt=attempt + 1,
                        retry_in_seconds=wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise ServiceRequestError(f"Failed to connect to service: {e}")

        raise last_exception or ServiceRequestError("Max retries exceeded")

    def _parse_response(self, result: Dict) -> List[List[Dict[str, Any]]]:
        if "json_result" in result and len(result["json_result"]) > 0:
            content = result["json_result"]
            if isinstance(content, list):
                all_results = content
            else:
                all_results = json.loads(content)

            if not isinstance(all_results, list):
                raise ValueError("响应格式错误：期望返回列表")

            for img_idx, img_result in enumerate(all_results):
                if not isinstance(img_result, list):
                    logger.warning(f"图片 {img_idx} 的结果格式错误，期望列表")
                    continue
                for block in img_result:
                    if not isinstance(block, dict):
                        continue
                    if "index" not in block:
                        block["index"] = 0
                    if "label" not in block:
                        block["label"] = "text"
                    if "bbox_2d" not in block:
                        block["bbox_2d"] = [0, 0, 0, 0]

            total_blocks = sum(len(img) for img in all_results)
            logger.info(
                "ocr_service_success",
                images=len(image_paths) if hasattr(self, 'image_paths') else 0,
                blocks=total_blocks,
            )
            return all_results

        raise ServiceResponseError(200, "Response missing 'json_result' field")

    async def process_images(
        self,
        image_paths: Union[str, List[str]],
        prompt: Optional[str] = None,
        custom_url: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        url = custom_url if custom_url is not None else self.service_url
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if self._rate_limiter and not self._rate_limiter.acquire(block=False):
            logger.warning("rate_limit_exceeded_queuing")
            await asyncio.sleep(0.1)

        if self._should_fallback():
            logger.warning("using_fallback_for_ocr", url=url)
            fallback = FallbackResult(page_num=1, reason="Circuit breaker open")
            return [[fallback.to_block()]]

        payload = self._create_multi_image_payload(image_paths)

        async def _do_request() -> List[List[Dict[str, Any]]]:
            async with httpx.AsyncClient(
                headers=self.async_headers,
                timeout=self.timeout,
                verify=False,
                limits=httpx.Limits(max_keepalive_connections=32, max_connections=128),
            ) as client:
                return await self._make_request_with_retry(client, url, payload)

        if self._circuit_breaker:
            try:
                return self._circuit_breaker.call(_do_request)
            except CircuitBreakerOpenError:
                logger.error("circuit_breaker_rejected_request")
                self._fallback_count += 1
                if self.fallback_enabled:
                    fallback = FallbackResult(page_num=1, reason="Circuit breaker open")
                    return [[fallback.to_block()]]
                raise ServiceRequestError("OCR service circuit breaker is open")
        else:
            return await _do_request()

    async def process_single_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        custom_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = await self.process_images(image_path, prompt, custom_url)
        return results[0] if results else []