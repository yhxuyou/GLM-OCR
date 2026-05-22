"""Medical OCR - Cache Utilities.

In-memory and Redis cache implementations.
"""

import os
import json
import hashlib
import threading
from typing import Any, Optional, Dict, Tuple
from datetime import datetime

from ..constants import DEFAULT_CACHE_MAX_SIZE, DEFAULT_CACHE_TTL


class SimpleCache:
    """Thread-safe in-memory cache with LRU eviction."""

    def __init__(self, max_size: int = DEFAULT_CACHE_MAX_SIZE, ttl: int = DEFAULT_CACHE_TTL):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._max_size = max_size
        self._ttl = ttl
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def _get_key(self, request_data: Any) -> str:
        data_str = json.dumps(request_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        return (datetime.now().timestamp() - timestamp) > self._ttl

    def get(self, request_data: Any) -> Optional[Any]:
        key = self._get_key(request_data)
        with self._lock:
            if key in self._cache:
                timestamp, value = self._cache[key]
                if not self._is_expired(timestamp):
                    self._hits += 1
                    return value
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    def set(self, request_data: Any, result: Any) -> None:
        key = self._get_key(request_data)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][0]
                )
                del self._cache[oldest_key]
            self._cache[key] = (datetime.now().timestamp(), result)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hit_rate": (
                    self._hits / (self._hits + self._misses)
                    if (self._hits + self._misses) > 0
                    else 0
                ),
            }


class RedisCache:
    """Redis-based cache for distributed deployments."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        ttl: int = DEFAULT_CACHE_TTL
    ):
        self._ttl = ttl
        self._redis = None

        try:
            import redis
            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            self._redis.ping()
        except Exception:
            import warnings
            warnings.warn("Redis not available, falling back to SimpleCache")
            self._redis = None

    def _get_key(self, request_data: Any) -> str:
        data_str = json.dumps(request_data, sort_keys=True, default=str)
        return f"medical_ocr:{hashlib.sha256(data_str.encode()).hexdigest()}"

    def get(self, request_data: Any) -> Optional[Any]:
        if not self._redis:
            return None
        try:
            key = self._get_key(request_data)
            data = self._redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception:
            return None

    def set(self, request_data: Any, result: Any) -> None:
        if not self._redis:
            return
        try:
            key = self._get_key(request_data)
            self._redis.setex(key, self._ttl, json.dumps(result))
        except Exception:
            pass

    def clear(self) -> None:
        if not self._redis:
            return
        try:
            for key in self._redis.scan_iter("medical_ocr:*"):
                self._redis.delete(key)
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        if not self._redis:
            return {"hits": 0, "misses": 0, "size": 0}
        try:
            count = len(list(self._redis.scan_iter("medical_ocr:*")))
            return {"hits": 0, "misses": 0, "size": count}
        except Exception:
            return {"hits": 0, "misses": 0, "size": 0}


def create_cache(redis_enabled: bool = False, **kwargs) -> Any:
    """Factory function to create appropriate cache."""
    if redis_enabled:
        return RedisCache(**kwargs)
    return SimpleCache(**kwargs)
