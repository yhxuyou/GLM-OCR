"""GLM-OCR Preprocessing Service Client.

Client library for interacting with the preprocessing microservice.
Handles HTTP communication, retry logic, and result parsing.

Usage:
    client = PreprocessClient("http://localhost:7001")
    result = client.preprocess_sync(image_path)
    # or async:
    job_id = client.preprocess_async(image_path)
    result = client.wait_for_result(job_id)
"""

from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, Optional

import requests

from PIL import Image


class PreprocessClient:
    """Client for the GLM-OCR preprocessing service."""

    def __init__(
        self,
        base_url: str = "http://localhost:7001",
        timeout: int = 30,
        retry_max_attempts: int = 3,
        retry_backoff: float = 1.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = retry_max_attempts
        self._retry_backoff = retry_backoff

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self._base_url}{endpoint}"
        for attempt in range(self._max_retries):
            try:
                response = requests.request(method, url, timeout=self._timeout, **kwargs)
                if response.status_code in (500, 502, 503, 504):
                    if attempt < self._max_retries - 1:
                        time.sleep(self._retry_backoff * (2**attempt))
                        continue
                return response
            except requests.exceptions.RequestException:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise
        raise RuntimeError("Max retries exceeded")

    def health(self) -> Dict[str, Any]:
        """Check service health."""
        response = self._request("GET", "/health")
        return response.json()

    def stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        response = self._request("GET", "/stats")
        return response.json()

    def _image_to_b64(self, image: Any) -> str:
        """Convert image to base64 encoded string."""
        if isinstance(image, str):
            if image.startswith(("http://", "https://", "file://")):
                return image
            try:
                with Image.open(image) as img:
                    buffer = io.BytesIO()
                    img.convert("RGB").save(buffer, format="JPEG", quality=90)
                    return base64.b64encode(buffer.getvalue()).decode("utf-8")
            except Exception:
                return image
        elif hasattr(image, "read"):
            img = Image.open(image).convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=90)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            raise ValueError("Unsupported image format")

    def preprocess_sync(self, image: Any) -> Dict[str, Any]:
        """Synchronous preprocessing - waits for result."""
        image_b64 = self._image_to_b64(image)
        payload = {"image": f"data:image/jpeg;base64,{image_b64}"}
        response = self._request("POST", "/preprocess/sync", json=payload)
        return response.json()

    def preprocess_async(self, image: Any) -> str:
        """Asynchronous preprocessing - returns job ID."""
        image_b64 = self._image_to_b64(image)
        payload = {"image": f"data:image/jpeg;base64,{image_b64}"}
        response = self._request("POST", "/preprocess", json=payload)
        data = response.json()
        return data["job_id"]

    def batch_preprocess(self, images: list) -> Dict[str, Any]:
        """Batch preprocessing."""
        b64_images = []
        for img in images:
            b64 = self._image_to_b64(img)
            b64_images.append(f"data:image/jpeg;base64,{b64}")
        payload = {"images": b64_images}
        response = self._request("POST", "/preprocess/batch", json=payload)
        return response.json()

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        response = self._request("GET", f"/status/{job_id}")
        return response.json()

    def get_result(self, job_id: str) -> Dict[str, Any]:
        """Get job result."""
        response = self._request("GET", f"/result/{job_id}")
        return response.json()

    def wait_for_result(
        self,
        job_id: str,
        timeout: int = 120,
        poll_interval: float = 1.0,
    ) -> Dict[str, Any]:
        """Wait for job completion and return result."""
        start = time.time()
        while time.time() - start < timeout:
            result = self.get_result(job_id)
            status = result.get("status")
            if status == "completed":
                return result
            if status == "failed":
                raise RuntimeError(f"Job failed: {result.get('error')}")
            time.sleep(poll_interval)
        raise TimeoutError(f"Job did not complete within {timeout} seconds")

    def preprocess(self, image: Any, sync: bool = True) -> Dict[str, Any]:
        """Unified preprocessing interface."""
        if sync:
            return self.preprocess_sync(image)
        job_id = self.preprocess_async(image)
        return self.wait_for_result(job_id)