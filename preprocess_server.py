"""GLM-OCR Image Preprocessing Microservice.

High-performance GPU-based image preprocessing service that handles:
    - Document Detection: Detect document boundaries in images
    - Orientation Detection: Determine document rotation (0°, 90°, 180°, 270°)
    - Deskew/Rectification: Correct document skew and perspective distortion

Architecture:
    - Flask HTTP layer for request handling
    - ThreadPoolExecutor with semaphore for controlled GPU concurrency
    - Job queue for backpressure handling
    - Health check and stats endpoints
    - Batch processing support

Usage:
    python preprocess_server.py --port 7001 --max-concurrent 4
    gunicorn -w 4 -k sync "preprocess_server:build_wsgi_app()"
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from flask import Flask, request, jsonify
    import numpy as np
    from PIL import Image, ImageOps
    import cv2
except ImportError as e:
    raise ImportError(f"Required packages missing: {e}\nInstall with: pip install flask numpy pillow opencv-python-headless")

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


class PreprocessStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PreprocessResult:
    document_detected: bool
    document_bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]
    orientation_angle: int = 0  # 0, 90, 180, 270
    deskewed_angle: float = 0.0
    corrected_image: Optional[str] = None  # base64 encoded
    confidence: float = 0.0


@dataclass
class PreprocessJob:
    job_id: str
    image_data: str  # base64 or file path
    status: PreprocessStatus = PreprocessStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[PreprocessResult] = None
    error_message: Optional[str] = None
    progress: float = 0.0


class DocumentDetector:
    """Document boundary detection using edge detection and contour analysis."""

    @staticmethod
    def detect(image: Image.Image) -> Tuple[bool, Optional[List[int]], float]:
        img = np.array(image)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, None, 0.0

        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        img_area = gray.shape[0] * gray.shape[1]

        if area < img_area * 0.05:
            return False, None, 0.0

        rect = cv2.boundingRect(max_contour)
        x, y, w, h = rect
        bbox = [x, y, x + w, y + h]

        confidence = min(area / img_area, 1.0)
        return True, bbox, confidence


class OrientationDetector:
    """Document orientation detection using text line analysis."""

    @staticmethod
    def detect(image: Image.Image) -> int:
        img = np.array(image)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10,
        )

        if lines is None or len(lines) == 0:
            return 0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

        if not angles:
            return 0

        angles = np.array(angles)
        angles = np.mod(angles, 180)

        bins = [0, 45, 90, 135, 180]
        counts, _ = np.histogram(angles, bins=bins)

        max_idx = np.argmax(counts)
        if max_idx == 0:
            return 0
        elif max_idx == 1:
            return 90
        elif max_idx == 2:
            return 180
        else:
            return 270


class DeskewCorrector:
    """Document deskewing using Hough line transform."""

    @staticmethod
    def correct(image: Image.Image, target_angle: Optional[int] = None) -> Tuple[Image.Image, float]:
        img = np.array(image)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        if target_angle is not None and target_angle != 0:
            img = np.array(Image.fromarray(img).rotate(-target_angle, expand=True))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=100,
            maxLineGap=10,
        )

        if lines is None or len(lines) == 0:
            return image, 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 45:
                    angles.append(angle)

        if not angles:
            return Image.fromarray(img), 0.0

        median_angle = np.median(angles)
        deskewed = Image.fromarray(img).rotate(-median_angle, expand=True)

        return deskewed, median_angle


class PreprocessWorkerPool:
    """Worker pool for GPU-based preprocessing with controlled concurrency."""

    def __init__(
        self,
        max_concurrent: int = 4,
        max_queue_size: int = 100,
    ):
        self._max_concurrent = max_concurrent
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._queue: queue.Queue[PreprocessJob] = queue.Queue(maxsize=max_queue_size)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="preprocess-wkr"
        )
        self._running = False
        self._stop_event = threading.Event()
        self._dispatcher: Optional[threading.Thread] = None

        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "active_workers": 0,
        }
        self._stats_lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="preprocess-dispatcher"
        )
        self._dispatcher.start()

    def stop(self, timeout: float = 30.0) -> None:
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._dispatcher:
            self._dispatcher.join(timeout=5)
        self._executor.shutdown(wait=True)

    def submit(self, job: PreprocessJob) -> bool:
        with self._stats_lock:
            self._stats["submitted"] += 1
        try:
            self._queue.put(job, timeout=1.0)
            return True
        except queue.Full:
            with self._stats_lock:
                self._stats["submitted"] -= 1
            return False

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            s = dict(self._stats)
        s["queue_size"] = self._queue.qsize()
        s["max_concurrent"] = self._max_concurrent
        s["max_queue_size"] = self._queue.maxsize
        return s

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._semaphore.acquire()
            with self._stats_lock:
                self._stats["active_workers"] = self._max_concurrent - self._semaphore._value

            job.status = PreprocessStatus.PROCESSING
            job.started_at = time.time()

            future = self._executor.submit(self._process_job, job)
            future.add_done_callback(lambda f, j=job: self._on_job_done(j, f))

    def _process_job(self, job: PreprocessJob) -> None:
        try:
            if job.image_data.startswith("data:image/"):
                data = job.image_data.split(",")[1]
                img_bytes = base64.b64decode(data)
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            elif job.image_data.startswith("file://"):
                image = Image.open(job.image_data[7:]).convert("RGB")
            elif job.image_data.startswith("http://") or job.image_data.startswith("https://"):
                import requests

                resp = requests.get(job.image_data, timeout=30)
                image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            else:
                image = Image.open(job.image_data).convert("RGB")

            job.progress = 25

            doc_detected, bbox, confidence = DocumentDetector.detect(image)
            job.progress = 50

            orientation = OrientationDetector.detect(image)
            job.progress = 75

            corrected_image, deskew_angle = DeskewCorrector.correct(image, orientation)

            buffer = io.BytesIO()
            corrected_image.save(buffer, format="JPEG", quality=90)
            corrected_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            job.result = PreprocessResult(
                document_detected=doc_detected,
                document_bbox=bbox,
                orientation_angle=orientation,
                deskewed_angle=deskew_angle,
                corrected_image=corrected_b64,
                confidence=confidence,
            )
            job.status = PreprocessStatus.COMPLETED
            job.progress = 100.0

        except Exception as e:
            job.status = PreprocessStatus.FAILED
            job.error_message = str(e)

    def _on_job_done(self, job: PreprocessJob, future: Future) -> None:
        self._semaphore.release()
        job.completed_at = time.time()
        with self._stats_lock:
            self._stats["active_workers"] = self._max_concurrent - self._semaphore._value
            if job.status == PreprocessStatus.COMPLETED:
                self._stats["completed"] += 1
            else:
                self._stats["failed"] += 1


class PreprocessService:
    """Preprocessing service with Flask HTTP endpoints."""

    def __init__(
        self,
        max_concurrent: int = 4,
        max_queue_size: int = 100,
        job_ttl_seconds: int = 1800,
    ):
        self._app = Flask(__name__)
        self._jobs: Dict[str, PreprocessJob] = {}
        self._jobs_lock = threading.Lock()
        self._job_ttl = job_ttl_seconds
        self._worker_pool = PreprocessWorkerPool(max_concurrent, max_queue_size)

        self._register_routes()

    def _register_routes(self) -> None:
        @self._app.route("/health", methods=["GET"])
        def health():
            stats = self._worker_pool.get_stats()
            return jsonify({"status": "ok", "stats": stats}), 200

        @self._app.route("/stats", methods=["GET"])
        def stats():
            stats = self._worker_pool.get_stats()
            with self._jobs_lock:
                job_counts = {s.value: 0 for s in PreprocessStatus}
                for job in self._jobs.values():
                    job_counts[job.status.value] += 1
            return jsonify({"pool": stats, "jobs": job_counts}), 200

        @self._app.route("/preprocess", methods=["POST"])
        def preprocess():
            if request.content_type != "application/json":
                return jsonify({"error": "Content-Type must be application/json"}), 400

            try:
                data = request.json
            except Exception:
                return jsonify({"error": "Invalid JSON"}), 400

            image_data = data.get("image")
            if not image_data:
                return jsonify({"error": "Missing 'image' field"}), 400

            job = PreprocessJob(job_id=uuid.uuid4().hex, image_data=image_data)

            with self._jobs_lock:
                self._jobs[job.job_id] = job

            accepted = self._worker_pool.submit(job)

            if not accepted:
                job.status = PreprocessStatus.FAILED
                job.error_message = "Queue is full"
                return (
                    jsonify({"error": "Queue full", "job_id": job.job_id, "retry_after": 5}),
                    503,
                )

            return (
                jsonify({
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "created_at": job.created_at,
                }),
                202,
            )

        @self._app.route("/preprocess/batch", methods=["POST"])
        def preprocess_batch():
            if request.content_type != "application/json":
                return jsonify({"error": "Content-Type must be application/json"}), 400

            try:
                data = request.json
            except Exception:
                return jsonify({"error": "Invalid JSON"}), 400

            images = data.get("images", [])
            if not images:
                return jsonify({"error": "Missing 'images' array"}), 400

            results = []
            for image_data in images[:50]:
                job = PreprocessJob(job_id=uuid.uuid4().hex, image_data=image_data)
                with self._jobs_lock:
                    self._jobs[job.job_id] = job
                accepted = self._worker_pool.submit(job)
                results.append({
                    "job_id": job.job_id,
                    "accepted": accepted,
                    "status": "pending" if accepted else "rejected",
                })

            accepted_count = sum(1 for r in results if r["accepted"])
            return (
                jsonify({
                    "results": results,
                    "accepted": accepted_count,
                    "rejected": len(results) - accepted_count,
                }),
                207 if accepted_count < len(results) else 200,
            )

        @self._app.route("/status/<job_id>", methods=["GET"])
        def status(job_id):
            with self._jobs_lock:
                job = self._jobs.get(job_id)

            if not job:
                return jsonify({"error": "Job not found"}), 404

            return jsonify({
                "job_id": job.job_id,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
            }), 200

        @self._app.route("/result/<job_id>", methods=["GET"])
        def result(job_id):
            with self._jobs_lock:
                job = self._jobs.get(job_id)

            if not job:
                return jsonify({"error": "Job not found"}), 404

            if job.status == PreprocessStatus.PENDING:
                return (
                    jsonify({"job_id": job_id, "status": "pending", "message": "Job queued"}),
                    202,
                )

            if job.status == PreprocessStatus.PROCESSING:
                return (
                    jsonify({
                        "job_id": job_id,
                        "status": "processing",
                        "progress": job.progress,
                    }),
                    202,
                )

            if job.status == PreprocessStatus.FAILED:
                return (
                    jsonify({"job_id": job_id, "status": "failed", "error": job.error_message}),
                    500,
                )

            if job.result:
                result_dict = {
                    "document_detected": job.result.document_detected,
                    "orientation_angle": job.result.orientation_angle,
                    "deskewed_angle": job.result.deskewed_angle,
                    "confidence": job.result.confidence,
                }
                if job.result.document_bbox:
                    result_dict["document_bbox"] = job.result.document_bbox
                if job.result.corrected_image:
                    result_dict["corrected_image"] = f"data:image/jpeg;base64,{job.result.corrected_image}"

                return (
                    jsonify({
                        "job_id": job_id,
                        "status": "completed",
                        "result": result_dict,
                        "completed_at": job.completed_at,
                    }),
                    200,
                )

            return jsonify({"error": "No result"}), 500

        @self._app.route("/preprocess/sync", methods=["POST"])
        def preprocess_sync():
            if request.content_type != "application/json":
                return jsonify({"error": "Content-Type must be application/json"}), 400

            try:
                data = request.json
            except Exception:
                return jsonify({"error": "Invalid JSON"}), 400

            image_data = data.get("image")
            if not image_data:
                return jsonify({"error": "Missing 'image' field"}), 400

            try:
                if image_data.startswith("data:image/"):
                    data_part = image_data.split(",")[1]
                    img_bytes = base64.b64decode(data_part)
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                elif image_data.startswith("file://"):
                    image = Image.open(image_data[7:]).convert("RGB")
                else:
                    image = Image.open(image_data).convert("RGB")

                doc_detected, bbox, confidence = DocumentDetector.detect(image)
                orientation = OrientationDetector.detect(image)
                corrected_image, deskew_angle = DeskewCorrector.correct(image, orientation)

                buffer = io.BytesIO()
                corrected_image.save(buffer, format="JPEG", quality=90)
                corrected_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                result_dict = {
                    "document_detected": doc_detected,
                    "orientation_angle": orientation,
                    "deskewed_angle": deskew_angle,
                    "confidence": confidence,
                    "corrected_image": f"data:image/jpeg;base64,{corrected_b64}",
                }
                if bbox:
                    result_dict["document_bbox"] = bbox

                return jsonify(result_dict), 200

            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def start(self) -> None:
        self._worker_pool.start()

    def stop(self) -> None:
        self._worker_pool.stop()

    @property
    def app(self) -> Flask:
        return self._app


def parse_args():
    parser = argparse.ArgumentParser(description="GLM-OCR Image Preprocessing Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7001, help="Port to bind to")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent GPU workers")
    parser.add_argument("--max-queue-size", type=int, default=100, help="Max queue size")
    parser.add_argument("--job-ttl", type=int, default=1800, help="Job TTL in seconds")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("GLM-OCR Image Preprocessing Service")
    print("=" * 60)
    print(f"  Host: {args.host}:{args.port}")
    print(f"  Max concurrent workers: {args.max_concurrent}")
    print(f"  Max queue size: {args.max_queue_size}")
    print(f"  Job TTL: {args.job_ttl}s")
    print("  Endpoints:")
    print("    POST /preprocess       - Submit preprocessing job (async)")
    print("    POST /preprocess/batch - Batch preprocessing")
    print("    POST /preprocess/sync  - Synchronous preprocessing")
    print("    GET  /status/<job_id>  - Check job status")
    print("    GET  /result/<job_id>  - Get job result")
    print("    GET  /health           - Health check")
    print("    GET  /stats            - Statistics")
    print("=" * 60)

    service = PreprocessService(
        max_concurrent=args.max_concurrent,
        max_queue_size=args.max_queue_size,
        job_ttl_seconds=args.job_ttl,
    )
    service.start()

    try:
        service.app.run(host=args.host, port=args.port, debug=False)
    finally:
        service.stop()


def build_wsgi_app() -> Flask:
    """Build WSGI app for gunicorn deployment."""
    service = PreprocessService()
    service.start()
    return service.app


if __name__ == "__main__":
    main()