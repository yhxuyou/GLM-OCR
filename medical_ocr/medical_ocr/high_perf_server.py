"""High-performance Medical OCR Server with async support.

Features:
- Async request handling using FastAPI
- Thread pool for CPU-bound OCR processing
- Redis-based caching for duplicate requests
- Batch processing support
- Prometheus metrics for monitoring
- Graceful shutdown handling
- Health check endpoints
"""

import asyncio
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import (
    start_http_server,
    Counter,
    Histogram,
    Gauge,
)

# Import MedicalOCR components
from medical_ocr import MedicalOcrPipeline
from glmocr.config import load_config

# Initialize FastAPI app
app = FastAPI(
    title="Medical OCR Server",
    description="High-performance OCR service for medical documents",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# Configuration
# =========================================================================

class Settings:
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    
    THREAD_POOL_SIZE: int = 8
    
    PROMETHEUS_PORT: int = 8001
    
    MAX_BATCH_SIZE: int = 50
    MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    MODEL_WARMUP_ENABLED: bool = True

settings = Settings()

# =========================================================================
# Metrics
# =========================================================================

REQUEST_COUNT = Counter(
    "ocr_requests_total",
    "Total number of OCR requests",
    ["endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "ocr_request_duration_seconds",
    "OCR request duration in seconds",
    ["endpoint"]
)

CACHE_HITS = Counter(
    "ocr_cache_hits_total",
    "Number of cache hits"
)

CACHE_MISSES = Counter(
    "ocr_cache_misses_total",
    "Number of cache misses"
)

ACTIVE_WORKERS = Gauge(
    "ocr_active_workers",
    "Number of active workers"
)

PENDING_TASKS = Gauge(
    "ocr_pending_tasks",
    "Number of pending tasks in queue"
)

# =========================================================================
# Redis Cache
# =========================================================================

class CacheManager:
    def __init__(self, host: str, port: int, db: int, ttl: int):
        self.client = redis.Redis(host=host, port=port, db=db)
        self.ttl = ttl
    
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key from request data."""
        if isinstance(data, dict):
            serialized = json.dumps(data, sort_keys=True).encode('utf-8')
        else:
            serialized = str(data).encode('utf-8')
        return hashlib.md5(serialized).hexdigest()
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached result."""
        try:
            value = self.client.get(key)
            if value:
                CACHE_HITS.inc()
                return json.loads(value)
            CACHE_MISSES.inc()
            return None
        except Exception:
            return None
    
    def set(self, key: str, value: Dict):
        """Set cache with TTL."""
        try:
            self.client.setex(key, self.ttl, json.dumps(value))
        except Exception:
            pass
    
    def delete(self, key: str):
        """Delete cached item."""
        try:
            self.client.delete(key)
        except Exception:
            pass
    
    def flush(self):
        """Flush all cache."""
        try:
            self.client.flushdb()
        except Exception:
            pass

# =========================================================================
# OCR Processor
# =========================================================================

class OCRProcessor:
    _instance = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self.pipeline = None
        self.executor = ThreadPoolExecutor(max_workers=settings.THREAD_POOL_SIZE)
    
    @classmethod
    async def get_instance(cls):
        """Get singleton instance with lazy initialization."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = OCRProcessor()
                    await cls._instance._initialize()
        return cls._instance
    
    async def _initialize(self):
        """Initialize pipeline in background."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._init_pipeline)
    
    def _init_pipeline(self):
        """Initialize MedicalOcrPipeline."""
        try:
            config = load_config()
            self.pipeline = MedicalOcrPipeline(
                config=config.pipeline,
                yolo_model_dir=getattr(config, 'yolo_model_dir', None),
                uvdoc_model_dir=getattr(config, 'uvdoc_model_dir', None),
            )
            self.pipeline.start()
            print(f"✅ OCR Pipeline initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize OCR Pipeline: {e}")
            raise
    
    async def process(self, request_data: Dict) -> Dict:
        """Process OCR request asynchronously."""
        if self.pipeline is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        loop = asyncio.get_event_loop()
        ACTIVE_WORKERS.inc()
        
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._process_sync,
                request_data
            )
            return result
        finally:
            ACTIVE_WORKERS.dec()
    
    def _process_sync(self, request_data: Dict) -> Dict:
        """Synchronous OCR processing."""
        results = list(self.pipeline.process(request_data))
        
        if len(results) == 1:
            return {
                "json_result": results[0].json_result,
                "markdown_result": results[0].markdown_result,
                "pages": 1
            }
        
        # Multiple pages
        return {
            "json_result": [r.json_result for r in results],
            "markdown_result": "\n\n---\n\n".join(r.markdown_result or "" for r in results),
            "pages": len(results)
        }
    
    async def process_batch(self, requests: List[Dict]) -> List[Dict]:
        """Process batch requests."""
        results = []
        
        for i, request_data in enumerate(requests):
            try:
                result = await self.process(request_data)
                results.append({
                    "index": i,
                    "success": True,
                    "data": result
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def shutdown(self):
        """Cleanup resources."""
        if self.pipeline:
            self.pipeline.stop()
        self.executor.shutdown(wait=True)

# =========================================================================
# Request Models
# =========================================================================

class OCRRequest(BaseModel):
    images: Union[str, List[str]] = Field(
        ..., description="Single image URL or list of image URLs"
    )
    options: Optional[Dict[str, Any]] = Field(
        None, description="Processing options"
    )

class BatchOCRRequest(BaseModel):
    requests: List[OCRRequest] = Field(
        ..., max_items=settings.MAX_BATCH_SIZE,
        description="List of OCR requests"
    )

class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status: pending/processing/completed/failed")
    created_at: datetime = Field(..., description="Task creation time")

class OCRResponse(BaseModel):
    json_result: str = Field(..., description="JSON formatted OCR result")
    markdown_result: str = Field(..., description="Markdown formatted OCR result")
    pages: int = Field(..., description="Number of processed pages")
    cached: bool = Field(False, description="Whether result was from cache")
    processing_time: float = Field(..., description="Processing time in seconds")

# =========================================================================
# Routes
# =========================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global cache_manager
    cache_manager = CacheManager(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        ttl=settings.REDIS_CACHE_TTL
    )
    
    # Initialize OCR processor
    await OCRProcessor.get_instance()
    
    # Start Prometheus metrics server
    start_http_server(settings.PROMETHEUS_PORT)
    
    print(f"🚀 Medical OCR Server started successfully")
    print(f"📊 Metrics available on port {settings.PROMETHEUS_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    processor = await OCRProcessor.get_instance()
    processor.shutdown()
    print("🛑 Medical OCR Server shut down gracefully")

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    processor = await OCRProcessor.get_instance()
    return {
        "status": "healthy",
        "service": "medical-ocr",
        "version": "1.0.0",
        "pipeline_ready": processor.pipeline is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics", tags=["Metrics"])
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest
    return generate_latest()

@app.post("/ocr/parse", tags=["OCR"], response_model=OCRResponse)
@REQUEST_LATENCY.time(endpoint="parse")
async def parse_document(request: OCRRequest):
    """Process a single OCR request."""
    start_time = datetime.now()
    
    # Normalize request data
    request_data = {
        "messages": [{
            "role": "user",
            "content": []
        }]
    }
    
    images = request.images
    if isinstance(images, str):
        images = [images]
    
    for image_url in images:
        request_data["messages"][0]["content"].append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })
    
    # Check cache
    cache_key = cache_manager.get_cache_key(request_data)
    cached_result = cache_manager.get(cache_key)
    
    if cached_result:
        processing_time = (datetime.now() - start_time).total_seconds()
        REQUEST_COUNT.labels(endpoint="parse", status="success").inc()
        
        return {
            **cached_result,
            "cached": True,
            "processing_time": processing_time
        }
    
    # Process request
    processor = await OCRProcessor.get_instance()
    result = await processor.process(request_data)
    
    # Cache result
    cache_manager.set(cache_key, result)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    REQUEST_COUNT.labels(endpoint="parse", status="success").inc()
    
    return {
        **result,
        "cached": False,
        "processing_time": processing_time
    }

@app.post("/ocr/batch", tags=["OCR"])
@REQUEST_LATENCY.time(endpoint="batch")
async def batch_parse(request: BatchOCRRequest):
    """Process batch OCR requests."""
    start_time = datetime.now()
    
    # Convert requests to internal format
    internal_requests = []
    for req in request.requests:
        data = {
            "messages": [{
                "role": "user",
                "content": []
            }]
        }
        
        images = req.images
        if isinstance(images, str):
            images = [images]
        
        for image_url in images:
            data["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        internal_requests.append(data)
    
    # Process batch
    processor = await OCRProcessor.get_instance()
    results = await processor.process_batch(internal_requests)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    REQUEST_COUNT.labels(endpoint="batch", status="success").inc()
    
    return {
        "results": results,
        "total": len(results),
        "success_count": sum(1 for r in results if r["success"]),
        "failed_count": sum(1 for r in results if not r["success"]),
        "processing_time": processing_time
    }

@app.delete("/cache", tags=["Cache"])
async def clear_cache():
    """Clear all cached results."""
    cache_manager.flush()
    return {"message": "Cache cleared successfully"}

@app.get("/cache/stats", tags=["Cache"])
async def cache_stats():
    """Get cache statistics."""
    return {
        "hits": CACHE_HITS._value.get(),
        "misses": CACHE_MISSES._value.get()
    }

# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        workers=4,
        loop="uvloop",
        reload=False
    )
