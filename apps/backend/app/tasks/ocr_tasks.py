"""OCR task definitions for Celery async processing."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import httpx

from app.tasks.celery_app import celery_app
from app.utils.logger import logger


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    name="ocr.process_image",
)
def process_image_task(
    self,
    image_path: str,
    ocr_url: str,
    task_id: str,
    page_num: int,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Process a single image through the OCR pipeline asynchronously.

    Args:
        image_path: Absolute path to the image file.
        ocr_url: GLM-OCR pipeline service URL.
        task_id: Parent task ID for traceability.
        page_num: Page number for ordering.
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys: page_num, success, result (or error).
    """
    logger.info(
        "ocr_task_started",
        task_id=task_id,
        page=page_num,
        image=image_path,
        attempt=self.request.retries + 1,
    )
    start = time.time()

    try:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(path, "rb") as f:
            import base64
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        ext = path.suffix.lower()
        mime = {
            ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp",
        }.get(ext, "image/png")
        data_uri = f"data:{mime};base64,{img_b64}"

        payload = {"images": [data_uri]}

        with httpx.Client(timeout=timeout, verify=False) as client:
            response = client.post(
                ocr_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"OCR service returned {response.status_code}: {response.text[:500]}"
            )

        result = response.json()

        latency = time.time() - start
        logger.info(
            "ocr_task_completed",
            task_id=task_id,
            page=page_num,
            latency_ms=round(latency * 1000),
        )

        return {
            "page_num": page_num,
            "success": True,
            "result": result,
            "latency_ms": round(latency * 1000),
        }

    except Exception as exc:
        latency = time.time() - start
        logger.error(
            "ocr_task_failed",
            task_id=task_id,
            page=page_num,
            error=str(exc),
            latency_ms=round(latency * 1000),
        )

        try:
            raise self.retry(exc=exc)
        except Exception:
            return {
                "page_num": page_num,
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "latency_ms": round(latency * 1000),
            }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    name="ocr.process_pages",
)
def process_pages_task(
    self,
    image_paths: List[str],
    ocr_url: str,
    task_id: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Process multiple pages as a group. Each page is a subtask.

    Uses Celery canvas (group) for parallel page processing.

    Args:
        image_paths: List of image file paths.
        ocr_url: GLM-OCR pipeline service URL.
        task_id: Parent task ID.
        timeout: Per-image timeout.

    Returns:
        Aggregated results dict.
    """
    from celery import group

    logger.info(
        "pages_task_started",
        task_id=task_id,
        total_pages=len(image_paths),
    )

    subtasks = [
        process_image_task.s(
            image_path=img_path,
            ocr_url=ocr_url,
            task_id=task_id,
            page_num=idx + 1,
            timeout=timeout,
        )
        for idx, img_path in enumerate(image_paths)
    ]

    job = group(subtasks)
    result = job.apply_async()

    try:
        page_results = result.get(timeout=len(image_paths) * timeout + 60)
    except Exception as exc:
        logger.error("pages_task_group_failed", task_id=task_id, error=str(exc))
        raise self.retry(exc=exc)

    successful = [r for r in page_results if r.get("success")]
    failed = [r for r in page_results if not r.get("success")]

    merged = _merge_page_results(successful)

    total_latency = sum(r.get("latency_ms", 0) for r in page_results)

    logger.info(
        "pages_task_completed",
        task_id=task_id,
        total_pages=len(image_paths),
        successful=len(successful),
        failed=len(failed),
        total_latency_ms=total_latency,
    )

    return {
        "success": len(failed) == 0,
        "total_pages": len(image_paths),
        "successful_pages": len(successful),
        "failed_pages": len(failed),
        "results": merged,
        "page_results": page_results,
        "total_latency_ms": total_latency,
    }


def _merge_page_results(page_results: List[Dict]) -> Dict[str, Any]:
    """Merge individual page results into a single document."""
    all_json = []
    all_markdown = []

    for pr in sorted(page_results, key=lambda x: x.get("page_num", 0)):
        result = pr.get("result", {})
        json_res = result.get("json_result", [])
        md_res = result.get("markdown_result", "")

        if isinstance(json_res, list):
            all_json.extend(json_res)

        if md_res:
            all_markdown.append(md_res)

    return {
        "json_result": all_json,
        "markdown_result": "\n\n---\n\n".join(all_markdown),
    }