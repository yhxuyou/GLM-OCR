"""Async OCR result aggregation backed by Redis.

The :class:`RegionAggregator` coordinates distributed OCR workers by tracking
per-document progress and collecting per-region results in Redis.  Workers
call :meth:`on_region_complete` as soon as a region has been processed; once
all regions are done the aggregated result can be retrieved via
:meth:`get_result`.

Redis layout
------------

``{prefix}:doc:{doc_id}:meta``
    Hash storing document metadata:

    * ``total_regions`` – expected number of regions
    * ``completed``     – number of regions processed so far
    * ``status``        – one of ``pending`` / ``processing`` / ``complete``

``{prefix}:doc:{doc_id}:regions``
    Hash mapping ``region_id`` -> JSON-encoded region result.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)

# Default TTL applied to completed documents (24 hours).
_DEFAULT_TTL_SECONDS = 60 * 60 * 24


class RegionAggregator:
    """Aggregate OCR region results from distributed async workers.

    Parameters
    ----------
    redis_url:
        Redis connection URL, e.g. ``redis://localhost:6379/0``.
    key_prefix:
        Prefix used for all Redis keys created by this aggregator.
    max_connections:
        Maximum number of connections kept in the async connection pool.
    """

    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "glmocr",
        max_connections: int = 10,
    ) -> None:
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._max_connections = max_connections

        self._pool: Optional[aioredis.ConnectionPool] = None
        self._redis: Optional[aioredis.Redis] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialise the Redis async connection pool."""
        if self._pool is not None:
            return

        self._pool = aioredis.ConnectionPool.from_url(
            self._redis_url,
            max_connections=self._max_connections,
            decode_responses=True,
        )
        self._redis = aioredis.Redis(connection_pool=self._pool)
        logger.debug(
            "RegionAggregator connected to Redis at %s (pool=%d)",
            self._redis_url,
            self._max_connections,
        )

    async def disconnect(self) -> None:
        """Close the Redis connection pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None
        logger.debug("RegionAggregator disconnected from Redis")

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _meta_key(self, doc_id: str) -> str:
        return f"{self._key_prefix}:doc:{doc_id}:meta"

    def _regions_key(self, doc_id: str) -> str:
        return f"{self._key_prefix}:doc:{doc_id}:regions"

    def _ensure_connected(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError(
                "RegionAggregator is not connected; call await connect() first"
            )
        return self._redis

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register_document(self, doc_id: str, total_regions: int, *, status: str = "processing") -> None:
        """Register a new document to be aggregated.

        Parameters
        ----------
        doc_id:
            Unique identifier of the document.
        total_regions:
            Total number of regions the document has been split into.
            Use 0 when the region count is not yet known (status must be "pending").
        status:
            Initial status, either "pending" (region count unknown) or "processing".
        """
        if total_regions < 0:
            raise ValueError("total_regions must be a non-negative integer")
        if total_regions == 0 and status != "pending":
            raise ValueError("total_regions=0 is only allowed when status='pending'")

        r = self._ensure_connected()
        meta_key = self._meta_key(doc_id)

        async with r.pipeline(transaction=True) as pipe:
            pipe.hset(
                meta_key,
                mapping={
                    "total_regions": str(total_regions),
                    "completed": "0",
                    "status": status,
                },
            )
            # Clean up any stale region hash from a previous attempt.
            pipe.delete(self._regions_key(doc_id))
            await pipe.execute()

        logger.info(
            "Registered document %s with %d region(s), status=%s",
            doc_id, total_regions, status,
        )

    async def on_region_complete(
        self, doc_id: str, region_id: str, result: Dict[str, Any]
    ) -> None:
        """Record a completed region and update progress atomically.

        Parameters
        ----------
        doc_id:
            Document identifier.
        region_id:
            Region identifier within the document.
        result:
            JSON-serialisable region result payload.
        """
        r = self._ensure_connected()
        meta_key = self._meta_key(doc_id)
        regions_key = self._regions_key(doc_id)

        try:
            serialized = json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.error(
                "Failed to serialise result for doc=%s region=%s: %s",
                doc_id,
                region_id,
                exc,
            )
            raise

        async with r.pipeline(transaction=True) as pipe:
            # Store the region result.
            pipe.hset(regions_key, region_id, serialized)
            # Atomically bump the completed counter.
            pipe.hincrby(meta_key, "completed", 1)
            await pipe.execute()

        # After the atomic block, check whether the document is now complete
        # so we can flip its status to "complete" exactly once.
        completed_raw, total_raw = await r.hmget(
            meta_key, "completed", "total_regions"
        )
        if completed_raw is None or total_raw is None:
            logger.warning(
                "on_region_complete called for unknown document %s", doc_id
            )
            return

        completed = int(completed_raw)
        total = int(total_raw)

        if completed >= total:
            await r.hset(meta_key, "status", "complete")
            logger.info(
                "Document %s complete (%d/%d regions)", doc_id, completed, total
            )
        else:
            logger.debug(
                "Document %s progress: %d/%d regions", doc_id, completed, total
            )

    async def get_progress(self, doc_id: str) -> Dict[str, Any]:
        """Return the current aggregation progress for a document.

        Returns
        -------
        dict
            ``{"completed": int, "total": int, "status": str}``.  If the
            document has not been registered, ``status`` is ``"unknown"`` and
            the counters are ``0``.
        """
        r = self._ensure_connected()
        meta_key = self._meta_key(doc_id)

        total_raw, completed_raw, status = await r.hmget(
            meta_key, "total_regions", "completed", "status"
        )

        if total_raw is None or completed_raw is None:
            return {"completed": 0, "total": 0, "status": "unknown"}

        return {
            "completed": int(completed_raw),
            "total": int(total_raw),
            "status": status or "unknown",
        }

    async def is_complete(self, doc_id: str) -> bool:
        """Return ``True`` if all regions for ``doc_id`` have been processed."""
        progress = await self.get_progress(doc_id)
        return (
            progress["status"] != "unknown"
            and progress["total"] > 0
            and progress["completed"] >= progress["total"]
        )

    async def get_result(self, doc_id: str) -> Dict[str, Any]:
        """Retrieve all region results for a completed document.

        The returned mapping is keyed by ``region_id`` and values are the
        deserialised region result dictionaries.

        After retrieval, a TTL is applied to the document's keys so that
        completed entries are automatically cleaned up.
        """
        r = self._ensure_connected()
        meta_key = self._meta_key(doc_id)
        regions_key = self._regions_key(doc_id)

        progress = await self.get_progress(doc_id)
        if progress["status"] == "unknown":
            raise KeyError(f"Document {doc_id!r} is not registered")

        regions_raw: Dict[str, str] = await r.hgetall(regions_key)

        results: Dict[str, Any] = {}
        for region_id, payload in regions_raw.items():
            try:
                results[region_id] = json.loads(payload)
            except json.JSONDecodeError as exc:
                logger.error(
                    "Failed to decode result for doc=%s region=%s: %s",
                    doc_id,
                    region_id,
                    exc,
                )
                # Keep the raw payload so the caller can still inspect it.
                results[region_id] = {"_raw": payload, "_decode_error": str(exc)}

        # Auto-cleanup: apply a TTL once the document is complete so stale
        # entries do not accumulate in Redis indefinitely.
        if await self.is_complete(doc_id):
            await r.expire(meta_key, _DEFAULT_TTL_SECONDS)
            await r.expire(regions_key, _DEFAULT_TTL_SECONDS)
            logger.debug(
                "Applied TTL of %ds to completed document %s",
                _DEFAULT_TTL_SECONDS,
                doc_id,
            )

        return results
