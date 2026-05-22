"""High Performance Medical OCR Server - Optimized Version.

This solves the three key bottlenecks:
1. 🔴 Pipeline singleton + thread pool → PipelinePool
2. 🔴 YOLO detection → Batch inference + GPU isolation
3. 🔴 Python GIL → ProcessPoolExecutor

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI (AsyncIO)                        │
│                      ↓                                       │
│               PipelinePool (N workers)                     │
│      ┌───────────────┐ ┌───────────────┐ ┌───────────────┐│
│      │  Worker 1     │ │  Worker 2     │ │  Worker N     ││
│      │  (GPU 0)      │ │  (GPU 1)      │ │  (GPU ...)    ││
│      │  Pipeline 1   │ │  Pipeline 2   │ │  Pipeline N   ││
│      └───────────────┘ └───────────────┘ └───────────────┘│
└─────────────────────────────────────────────────────────────┘
"""

import os
import sys
import time
import uuid
import json
import hashlib
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project path
sys.path.insert(0, '/workspace/medical_ocr')

# Import our pipeline pool
from medical_ocr.pipeline_pool import PipelinePool, PipelinePoolConfig, create_pipeline_pool
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)

# =========================================================================
# Global State
# =========================================================================

# The pipeline pool (initialized on startup)
pipeline_pool: Optional[PipelinePool] = None

# =========================================================================
# Data Models
# =========================================================================

class OCRRequest(BaseModel):
    """Single OCR request."""
    images: List[str] = Field(..., description="List of image URLs or paths")
    options: Optional[Dict[str, Any]] = Field(None, description="Processing options")

class OCRBatchRequest(BaseModel):
    """Batch OCR request."""
    requests: List[OCRRequest] = Field(..., description="List of OCR requests")

class OCRResponse(BaseModel):
    """OCR response."""
    request_id: str = Field(..., description="Unique request ID")
    json_result: Optional[Any] = Field(None, description="JSON-formatted OCR result")
    markdown_result: Optional[str] = Field(None, description="Markdown-formatted OCR result")
    processing_time: float = Field(..., description="Processing time in seconds")
    cached: bool = Field(False, description="Whether result was cached")
    worker_idx: Optional[int] = Field(None, description="Worker index that processed this")
    gpu_id: Optional[int] = Field(None, description="GPU ID that processed this")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    pool_size: int
    gpu_enabled: bool
    cache_hits: int
    cache_misses: int

# =========================================================================
# Simple Cache (Redis optional fallback)
# =========================================================================

class SimpleCache:
    """Simple in-memory cache with optional Redis backend."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
    
    def _get_key(self, request_data: Any) -> str:
        """Generate cache key from request data."""
        data_str = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def get(self, request_data: Any) -> Optional[Any]:
        """Get cached result."""
        key = self._get_key(request_data)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key][1]
            self._misses += 1
            return None
    
    def set(self, request_data: Any, result: Any, ttl: int = 3600):
        """Set cached result."""
        key = self._get_key(request_data)
        with self._lock:
            if len(self._cache) >= self._max_size:
                # Simple LRU: remove oldest
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), result)
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                'hits': self._hits,
                'misses': self._misses,
                'size': len(self._cache)
            }

# Create global cache instance
cache = SimpleCache()

# =========================================================================
# FastAPI Application
# =========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    # Startup
    global pipeline_pool
    logger.info("Starting Medical OCR High Performance Server")
    
    try:
        # Initialize pipeline pool
        # Replace with actual config loading
        from glmocr.config import load_config
        config = load_config()
        
        pool_size = int(os.getenv("POOL_SIZE", "4"))
        use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
        use_processes = os.getenv("USE_PROCESSES", "true").lower() == "true"
        
        # Parse GPU IDs if specified
        gpu_ids_str = os.getenv("GPU_IDS", None)
        gpu_ids = [int(x) for x in gpu_ids_str.split(",")] if gpu_ids_str else None
        
        logger.info(f"Creating pipeline pool with size {pool_size}, GPU: {use_gpu}")
        
        pipeline_pool = create_pipeline_pool(
            pipeline_config=config.pipeline,
            pool_size=pool_size,
            yolo_model_dir=os.getenv("YOLO_MODEL_DIR", None),
            uvdoc_model_dir=os.getenv("UVDOC_MODEL_DIR", None),
            use_processes=use_processes,
            use_gpu=use_gpu,
            gpu_ids=gpu_ids
        )
        
        # Initialize the pool
        pipeline_pool.initialize()
        
        logger.info("✅ Pipeline pool initialized successfully")
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Medical OCR Server")
        if pipeline_pool:
            try:
                pipeline_pool.shutdown()
            except Exception:
                pass
        logger.info("Server shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Medical OCR High Performance Server",
    description="High-performance OCR service for medical documents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# API Routes
# =========================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "medical-ocr-high-performance",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "parse": "/ocr/parse",
            "batch": "/ocr/batch",
            "cache": "/cache/stats"
        }
    }

@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    cache_stats = cache.stats()
    return HealthResponse(
        status="healthy" if pipeline_pool else "initializing",
        timestamp=datetime.utcnow().isoformat(),
        pool_size=pipeline_pool.config.pool_size if pipeline_pool else 0,
        gpu_enabled=pipeline_pool.config.use_gpu if pipeline_pool else False,
        cache_hits=cache_stats['hits'],
        cache_misses=cache_stats['misses']
    )

@app.post("/ocr/parse", tags=["OCR"], response_model=OCRResponse)
async def parse_document(request: OCRRequest):
    """Process a single OCR request with high performance pipeline pool.
    
    This endpoint uses the PipelinePool for parallel processing across
    multiple GPU-isolated workers.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Build request data
    request_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in request.images
                ]
            }
        ]
    }
    
    # Check cache first
    cached_result = cache.get(request_data)
    if cached_result is not None:
        processing_time = time.time() - start_time
        return OCRResponse(
            request_id=request_id,
            json_result=cached_result.get('json_result'),
            markdown_result=cached_result.get('markdown_result'),
            processing_time=processing_time,
            cached=True
        )
    
    # Process through pipeline pool
    if not pipeline_pool:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    try:
        # Submit to pool
        result = pipeline_pool.process(request_data)
        
        # Check if worker returned an error
        if isinstance(result, dict) and 'error' in result:
            logger.error(f"Worker error: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Extract first result
        if result and isinstance(result, list) and len(result) > 0:
            single_result = result[0]
            
            # Cache result
            cache.set(request_data, single_result)
            
            processing_time = time.time() - start_time
            
            return OCRResponse(
                request_id=request_id,
                json_result=single_result.get('json_result'),
                markdown_result=single_result.get('markdown_result'),
                processing_time=processing_time,
                cached=False,
                worker_idx=single_result.get('worker_idx'),
                gpu_id=single_result.get('gpu_id')
            )
        
        # Fallback
        raise HTTPException(status_code=500, detail="Pipeline returned empty result")
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/batch", tags=["OCR"])
async def parse_batch(request: OCRBatchRequest):
    """Process batch OCR requests in parallel.
    
    Each request in the batch is handled by a separate worker in the pool.
    """
    start_time = time.time()
    
    # Build individual requests
    request_datas = []
    for req in request.requests:
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}}
                        for url in req.images
                    ]
                }
            ]
        }
        request_datas.append(request_data)
    
    # Process batch in parallel
    if not pipeline_pool:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    try:
        results = pipeline_pool.process_batch(request_datas)
        
        total_time = time.time() - start_time
        
        return {
            "batch_id": str(uuid.uuid4()),
            "results": results,
            "total_time": total_time,
            "request_count": len(request.requests)
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats", tags=["Cache"])
async def cache_stats():
    """Get cache statistics."""
    return cache.stats()

@app.delete("/cache", tags=["Cache"])
async def clear_cache():
    """Clear all cached results."""
    global cache
    cache = SimpleCache()
    return {"message": "Cache cleared successfully"}

# =========================================================================
# Server Startup
# =========================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run server with multiple workers
    # Note: In production, you should use a proper ASGI server
    # like gunicorn + uvicorn worker
    
    uvicorn.run(
        "medical_ocr.high_perf_server_v2:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        limit_concurrency=100,
        access_log=True
    )
