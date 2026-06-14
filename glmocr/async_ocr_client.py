"""Asynchronous OCR client using httpx.AsyncClient."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import traceback
import random
from typing import TYPE_CHECKING, Dict, Tuple, Optional, List
from urllib.parse import urlparse

import httpx

from glmocr.utils.logging import get_logger, get_profiler

if TYPE_CHECKING:
    from glmocr.config import OCRApiConfig

logger = get_logger(__name__)
profiler = get_profiler(__name__)


class AsyncOCRClient:
    """
    Asynchronous OCR client that calls a remote API for recognition.
    Uses httpx.AsyncClient for connection pooling and async HTTP requests.
    """

    def __init__(self, config: "OCRApiConfig"):
        """Initialize the async OCR client.

        Args:
            config: OCRApiConfig instance.
        """
        self.config = config

        # API service configuration
        self.api_host = config.api_host
        self.api_port = config.api_port

        self.api_scheme = config.api_scheme
        if not self.api_scheme:
            self.api_scheme = "https" if self.api_port == 443 else "http"

        self.api_path = config.api_path
        if not str(self.api_path).startswith("/"):
            self.api_path = f"/{self.api_path}"

        self.api_url = config.api_url
        if not self.api_url:
            self.api_url = (
                f"{self.api_scheme}://{self.api_host}:{self.api_port}{self.api_path}"
            )

        self.api_key = (
            config.api_key
            or os.getenv("ZHIPU_API_KEY")
            or os.getenv("GLMOCR_API_KEY")  # legacy fallback
        )
        self.extra_headers = config.headers or {}

        # API mode: "openai" or "ollama_generate"
        self.api_mode = getattr(config, "api_mode", "openai")

        # SSL verification
        self.verify_ssl = config.verify_ssl

        # If api_url is set, try to infer host/port for the connect() preflight
        try:
            parsed = urlparse(self.api_url)
            if parsed.hostname:
                self.api_host = parsed.hostname
            if parsed.port:
                self.api_port = parsed.port
        except Exception:
            pass

        # Connection timeout configuration
        self.connect_timeout = config.connect_timeout
        self.request_timeout = config.request_timeout

        # Retry configuration
        self.retry_max_attempts = getattr(config, "retry_max_attempts", 2)
        self.retry_backoff_base_seconds = getattr(
            config, "retry_backoff_base_seconds", 0.5
        )
        self.retry_backoff_max_seconds = getattr(
            config, "retry_backoff_max_seconds", 8.0
        )
        self.retry_jitter_ratio = getattr(config, "retry_jitter_ratio", 0.2)
        self.retry_status_codes = set(
            getattr(config, "retry_status_codes", [429, 500, 502, 503, 504])
        )

        # Async client instance
        self._client: Optional[httpx.AsyncClient] = None

        # Connection pool size (from config, or default). Should be >= pipeline max_workers.
        pool_size = getattr(config, "connection_pool_size", None)
        self._pool_maxsize = pool_size if pool_size is not None else 128

        # Model information
        self.model = config.model

    async def start(self):
        """Initialize the async HTTP client with connection pooling."""
        logger.debug("Initializing async OCR client for %s...", self.api_url)
        
        if self._client is None:
            # Create httpx.AsyncClient with connection pooling
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self._pool_maxsize,
                    max_keepalive_connections=self._pool_maxsize,
                ),
                timeout=httpx.Timeout(self.request_timeout),
                verify=self.verify_ssl,
            )
        
        logger.debug("Async OCR client initialized successfully!")

    async def stop(self):
        """Close the async HTTP client and release resources."""
        logger.debug("Stopping async OCR client...")
        
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
        
        logger.debug("Async OCR client stopped.")

    async def is_alive(self, timeout: float = 5.0) -> bool:
        """Quick socket-level check whether the API port is still reachable.
        
        Args:
            timeout: Socket connection timeout in seconds.
            
        Returns:
            True if the API port is reachable, False otherwise.
        """
        try:
            # Use asyncio for non-blocking socket check
            loop = asyncio.get_event_loop()
            
            def _check_socket():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(timeout)
                    return sock.connect_ex((self.api_host, self.api_port)) == 0
            
            return await loop.run_in_executor(None, _check_socket)
        except Exception:
            return False

    async def _sleep_backoff(
        self, attempt_index: int, retry_after_seconds: Optional[float] = None
    ) -> None:
        """Async sleep with exponential backoff and jitter.
        
        Args:
            attempt_index: Current attempt index (0-based).
            retry_after_seconds: Optional Retry-After header value.
        """
        if retry_after_seconds is not None and retry_after_seconds > 0:
            sleep_s = min(
                float(retry_after_seconds), float(self.retry_backoff_max_seconds)
            )
        else:
            base = float(self.retry_backoff_base_seconds)
            sleep_s = min(
                base * (2**attempt_index), float(self.retry_backoff_max_seconds)
            )

        jitter = sleep_s * float(self.retry_jitter_ratio)
        if jitter > 0:
            sleep_s = max(0.0, sleep_s + random.uniform(-jitter, jitter))

        await asyncio.sleep(sleep_s)

    @staticmethod
    def _parse_retry_after_seconds(response: httpx.Response) -> Optional[float]:
        """Parse Retry-After header from response.
        
        Args:
            response: httpx.Response object.
            
        Returns:
            Retry-After value in seconds, or None if not present.
        """
        ra = response.headers.get("Retry-After")
        if not ra:
            return None
        try:
            return float(ra)
        except Exception:
            return None

    async def async_process(self, request_data: Dict) -> Tuple[Dict, int]:
        """
        Process a recognition request asynchronously.

        Args:
            request_data: Request payload containing messages, etc.

        Returns:
            (response_dict, http_status_code)
        """
        if self._client is None:
            await self.start()

        # Convert request format based on API mode
        if self.api_mode == "ollama_generate":
            request_data = self._convert_to_ollama_generate(request_data)
        else:
            # Inject model field if configured (required by Ollama/MLX)
            if self.model:
                request_data["model"] = self.model

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        total_attempts = int(self.retry_max_attempts) + 1
        last_error: Optional[str] = None

        for attempt in range(total_attempts):
            try:
                with profiler.measure("async_http_request"):
                    response = await self._client.post(
                        self.api_url,
                        headers=headers,
                        json=request_data,
                    )

                if response.status_code == 200:
                    result = response.json()

                    # Parse response based on API mode
                    if self.api_mode == "ollama_generate":
                        # Ollama generate format: {"response": "...", "done": true, ...}
                        # Check for error field first
                        if "error" in result:
                            error_msg = result.get("error", "Unknown error")
                            logger.error("Ollama API returned error: %s", error_msg)
                            return {"error": f"Ollama API error: {error_msg}"}, 500

                        # Extract response field
                        output = result.get("response")
                        if output is None:
                            logger.error(
                                "Ollama API response missing 'response' field. Response: %s",
                                str(result)[:500],
                            )
                            return {
                                "error": "Invalid Ollama API response format: missing 'response' field"
                            }, 500
                    else:
                        # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
                        try:
                            output = result["choices"][0]["message"]["content"]
                        except (KeyError, IndexError, TypeError) as e:
                            logger.error(
                                "Invalid OpenAI API response format: %s. Response: %s",
                                str(e),
                                str(result)[:500],
                            )
                            return {
                                "error": f"Invalid OpenAI API response format: {str(e)}"
                            }, 500

                    return {
                        "choices": [{"message": {"content": (output or "").strip()}}]
                    }, 200

                status = int(response.status_code)
                body_preview = (response.text or "")[:500]

                if status in self.retry_status_codes and attempt < total_attempts - 1:
                    retry_after = self._parse_retry_after_seconds(response)
                    logger.warning(
                        "Receive status %s from OCR API (attempt %d/%d). Retrying... response: %s",
                        status,
                        attempt + 1,
                        total_attempts,
                        body_preview,
                    )
                    await self._sleep_backoff(
                        attempt_index=attempt, retry_after_seconds=retry_after
                    )
                    continue

                logger.warning(
                    "Receive bad status code: %s, response: %s",
                    status,
                    body_preview,
                )
                return {
                    "error": "API request failed",
                    "status_code": status,
                    "response": body_preview,
                }, status

            except httpx.RequestError as e:
                last_error = str(e)
                if attempt < total_attempts - 1:
                    logger.warning(
                        "OCR API request error (attempt %d/%d): %s. Retrying...",
                        attempt + 1,
                        total_attempts,
                        last_error,
                    )
                    await self._sleep_backoff(attempt_index=attempt)
                    continue
                logger.error("Error during recognition: %s", last_error)
                logger.debug(traceback.format_exc())
                return {"error": f"Error during recognition: {last_error}"}, 500

            except Exception as e:
                logger.error("Error during recognition: %s", e)
                logger.debug(traceback.format_exc())
                return {"error": f"Error during recognition: {str(e)}"}, 500

        return {
            "error": f"API request failed after {total_attempts} attempts",
            "detail": last_error,
        }, 500

    async def async_process_batch(
        self, requests: List[Dict]
    ) -> List[Tuple[Dict, int]]:
        """
        Process multiple recognition requests concurrently.

        Args:
            requests: List of request payloads.

        Returns:
            List of (response_dict, http_status_code) tuples in the same order as input.
        """
        if not requests:
            return []

        # Use asyncio.gather to process all requests concurrently
        tasks = [self.async_process(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        return list(results)

    def _convert_to_ollama_generate(self, request_data: Dict) -> Dict:
        """Convert OpenAI chat format to Ollama generate format.

        OpenAI format:
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "..."},
                        {"type": "image_url", "image_url": "data:image/...;base64,..."}
                    ]
                }
            ],
            "max_tokens": 100,
            ...
        }

        Ollama generate format:
        {
            "model": "glm-ocr:latest",
            "prompt": "...",
            "images": ["base64_string"],
            "stream": false,
            "options": {
                "num_predict": 100,
                ...
            }
        }

        Note:
            This method only processes messages with role='user'. System messages
            and conversation history (assistant messages) are not supported by
            Ollama's /api/generate endpoint and will be ignored.
        """
        messages = request_data.get("messages", [])

        # Check for non-user messages and log a warning
        non_user_messages = [msg for msg in messages if msg.get("role") != "user"]
        if non_user_messages:
            roles = [msg.get("role") for msg in non_user_messages]
            logger.warning(
                f"Ollama generate mode: ignoring {len(non_user_messages)} non-user message(s) "
                f"with roles {roles}. Only the last user message will be processed."
            )

        # Extract prompt and images from the last user message
        prompt = ""
        images = []

        # Find the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg
                break

        if last_user_msg:
            content = last_user_msg.get("content", "")

            if isinstance(content, str):
                prompt = content
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        prompt = item.get("text", "")
                    elif item.get("type") == "image_url":
                        # Extract base64 from data URI
                        image_url = item.get("image_url", "")
                        if isinstance(image_url, dict):
                            image_url = image_url.get("url", "")

                        # Parse data:image/...;base64,<data>
                        if image_url.startswith("data:"):
                            parts = image_url.split(",", 1)
                            if len(parts) == 2:
                                images.append(parts[1])
                        else:
                            images.append(image_url)

        # Build Ollama generate request
        # Default to "Text Recognition:" when prompt is empty because Ollama
        # requires a non-empty prompt to trigger the model. This matches the
        # official documented usage for GLM-OCR on Ollama.
        ollama_request = {
            "model": self.model or "glm-ocr:latest",
            "prompt": prompt if prompt else "Text Recognition:",
            "stream": False,
        }

        if images:
            ollama_request["images"] = images

        # Map parameters to Ollama options
        options = {}
        if "max_tokens" in request_data:
            options["num_predict"] = request_data["max_tokens"]
        if "temperature" in request_data:
            options["temperature"] = request_data["temperature"]
        if "top_p" in request_data:
            options["top_p"] = request_data["top_p"]
        if "top_k" in request_data:
            options["top_k"] = request_data["top_k"]
        if "repetition_penalty" in request_data:
            options["repeat_penalty"] = request_data["repetition_penalty"]

        if options:
            ollama_request["options"] = options

        return ollama_request
