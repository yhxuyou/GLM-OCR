"""Unit tests for RegionAggregator (async Redis-backed result aggregation)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRegionAggregatorLifecycle:
    """Tests for RegionAggregator connect/disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_creates_pool_and_redis(self):
        """connect() initializes Redis connection pool and client."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
            max_connections=5,
        )

        with patch("glmocr.aggregator.aioredis.ConnectionPool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.from_url.return_value = mock_pool

            with patch("glmocr.aggregator.aioredis.Redis") as mock_redis_cls:
                mock_redis = MagicMock()
                mock_redis_cls.return_value = mock_redis

                await aggregator.connect()

                # Verify pool was created with correct parameters
                mock_pool_cls.from_url.assert_called_once_with(
                    "redis://localhost:6379/0",
                    max_connections=5,
                    decode_responses=True,
                )
                # Verify Redis client was created
                mock_redis_cls.assert_called_once_with(connection_pool=mock_pool)
                # Verify internal state
                assert aggregator._pool is mock_pool
                assert aggregator._redis is mock_redis

    @pytest.mark.asyncio
    async def test_connect_idempotent(self):
        """connect() is idempotent - calling twice doesn't recreate pool."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(redis_url="redis://localhost:6379/0")

        with patch("glmocr.aggregator.aioredis.ConnectionPool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.from_url.return_value = mock_pool

            with patch("glmocr.aggregator.aioredis.Redis"):
                await aggregator.connect()
                await aggregator.connect()

                # Should only create pool once
                assert mock_pool_cls.from_url.call_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_closes_connections(self):
        """disconnect() closes Redis client and connection pool."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(redis_url="redis://localhost:6379/0")

        mock_redis = AsyncMock()
        mock_pool = AsyncMock()
        aggregator._redis = mock_redis
        aggregator._pool = mock_pool

        await aggregator.disconnect()

        # Verify both were closed
        mock_redis.aclose.assert_called_once()
        mock_pool.aclose.assert_called_once()
        # Verify internal state cleared
        assert aggregator._redis is None
        assert aggregator._pool is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """disconnect() is safe to call when not connected."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(redis_url="redis://localhost:6379/0")

        # Should not raise
        await aggregator.disconnect()

    @pytest.mark.asyncio
    async def test_ensure_connected_raises_when_not_connected(self):
        """_ensure_connected() raises RuntimeError if not connected."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(redis_url="redis://localhost:6379/0")

        with pytest.raises(RuntimeError, match="not connected"):
            aggregator._ensure_connected()


class TestRegionAggregatorRegisterDocument:
    """Tests for RegionAggregator.register_document()."""

    @pytest.mark.asyncio
    async def test_register_document_sets_metadata(self):
        """register_document() creates metadata hash with correct fields."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value.__aenter__.return_value = mock_pipeline
        aggregator._redis = mock_redis

        await aggregator.register_document("doc123", total_regions=5)

        # Verify pipeline was used
        mock_redis.pipeline.assert_called_once_with(transaction=True)

        # Verify hset was called with correct metadata
        calls = mock_pipeline.hset.call_args_list
        assert len(calls) == 1
        meta_key, mapping = calls[0][0]
        assert meta_key == "test:doc:doc123:meta"
        assert mapping["total_regions"] == "5"
        assert mapping["completed"] == "0"
        assert mapping["status"] == "processing"

        # Verify old regions were deleted
        mock_pipeline.delete.assert_called_once_with("test:doc:doc123:regions")

        # Verify pipeline was executed
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_document_rejects_zero_regions(self):
        """register_document() raises ValueError for total_regions <= 0."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(redis_url="redis://localhost:6379/0")
        aggregator._redis = AsyncMock()

        with pytest.raises(ValueError, match="positive integer"):
            await aggregator.register_document("doc123", total_regions=0)

        with pytest.raises(ValueError, match="positive integer"):
            await aggregator.register_document("doc123", total_regions=-1)

    @pytest.mark.asyncio
    async def test_register_document_rejects_when_not_connected(self):
        """register_document() raises RuntimeError if not connected."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(redis_url="redis://localhost:6379/0")

        with pytest.raises(RuntimeError, match="not connected"):
            await aggregator.register_document("doc123", total_regions=5)


class TestRegionAggregatorOnRegionComplete:
    """Tests for RegionAggregator.on_region_complete()."""

    @pytest.mark.asyncio
    async def test_on_region_complete_stores_result_and_increments_counter(self):
        """on_region_complete() stores region result and increments completed counter."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value.__aenter__.return_value = mock_pipeline
        mock_redis.hmget.return_value = ("2", "5")  # completed=2, total=5
        aggregator._redis = mock_redis

        result = {"text": "Hello", "bbox": [10, 20, 100, 200]}
        await aggregator.on_region_complete("doc123", "region_0", result)

        # Verify region result was stored
        calls = mock_pipeline.hset.call_args_list
        assert len(calls) == 1
        regions_key, region_id, serialized = calls[0][0]
        assert regions_key == "test:doc:doc123:regions"
        assert region_id == "region_0"
        assert json.loads(serialized) == result

        # Verify counter was incremented
        mock_pipeline.hincrby.assert_called_once_with(
            "test:doc:doc123:meta", "completed", 1
        )

        # Verify pipeline was executed
        mock_pipeline.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_region_complete_sets_status_to_complete_when_done(self):
        """on_region_complete() sets status to 'complete' when all regions done."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value.__aenter__.return_value = mock_pipeline
        # Simulate last region: completed=5, total=5
        mock_redis.hmget.return_value = ("5", "5")
        aggregator._redis = mock_redis

        await aggregator.on_region_complete("doc123", "region_4", {"text": "last"})

        # Verify status was set to 'complete'
        mock_redis.hset.assert_called_once_with(
            "test:doc:doc123:meta", "status", "complete"
        )

    @pytest.mark.asyncio
    async def test_on_region_complete_does_not_set_complete_when_not_done(self):
        """on_region_complete() doesn't set status to 'complete' if more regions pending."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value.__aenter__.return_value = mock_pipeline
        # Simulate middle region: completed=2, total=5
        mock_redis.hmget.return_value = ("2", "5")
        aggregator._redis = mock_redis

        await aggregator.on_region_complete("doc123", "region_1", {"text": "middle"})

        # Verify status was NOT set to 'complete'
        mock_redis.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_region_complete_handles_unknown_document(self):
        """on_region_complete() handles unknown document gracefully."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value.__aenter__.return_value = mock_pipeline
        # Simulate unknown document: hmget returns None
        mock_redis.hmget.return_value = (None, None)
        aggregator._redis = mock_redis

        # Should not raise
        await aggregator.on_region_complete("unknown_doc", "region_0", {"text": "test"})

    @pytest.mark.asyncio
    async def test_on_region_complete_rejects_non_serializable_result(self):
        """on_region_complete() raises error for non-JSON-serializable results."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value.__aenter__.return_value = mock_pipeline
        aggregator._redis = mock_redis

        # Create a non-serializable object
        class NonSerializable:
            pass

        with pytest.raises((TypeError, ValueError)):
            await aggregator.on_region_complete(
                "doc123", "region_0", {"obj": NonSerializable()}
            )


class TestRegionAggregatorGetProgress:
    """Tests for RegionAggregator.get_progress()."""

    @pytest.mark.asyncio
    async def test_get_progress_returns_correct_fields(self):
        """get_progress() returns completed, total, and status."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_redis.hmget.return_value = ("3", "5", "processing")
        aggregator._redis = mock_redis

        progress = await aggregator.get_progress("doc123")

        assert progress == {
            "completed": 3,
            "total": 5,
            "status": "processing",
        }

        # Verify correct key was queried
        mock_redis.hmget.assert_called_once_with(
            "test:doc:doc123:meta", "total_regions", "completed", "status"
        )

    @pytest.mark.asyncio
    async def test_get_progress_returns_unknown_for_unregistered_document(self):
        """get_progress() returns status='unknown' for unregistered documents."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_redis.hmget.return_value = (None, None, None)
        aggregator._redis = mock_redis

        progress = await aggregator.get_progress("unknown_doc")

        assert progress == {
            "completed": 0,
            "total": 0,
            "status": "unknown",
        }

    @pytest.mark.asyncio
    async def test_get_progress_handles_missing_status(self):
        """get_progress() handles missing status field gracefully."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        # Status is None but counters exist
        mock_redis.hmget.return_value = ("5", "5", None)
        aggregator._redis = mock_redis

        progress = await aggregator.get_progress("doc123")

        assert progress["status"] == "unknown"
        assert progress["completed"] == 5
        assert progress["total"] == 5


class TestRegionAggregatorIsComplete:
    """Tests for RegionAggregator.is_complete()."""

    @pytest.mark.asyncio
    async def test_is_complete_returns_true_when_all_regions_done(self):
        """is_complete() returns True when completed >= total."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_redis.hmget.return_value = ("5", "5", "complete")
        aggregator._redis = mock_redis

        assert await aggregator.is_complete("doc123") is True

    @pytest.mark.asyncio
    async def test_is_complete_returns_false_when_not_done(self):
        """is_complete() returns False when completed < total."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_redis.hmget.return_value = ("3", "5", "processing")
        aggregator._redis = mock_redis

        assert await aggregator.is_complete("doc123") is False

    @pytest.mark.asyncio
    async def test_is_complete_returns_false_for_unknown_document(self):
        """is_complete() returns False for unregistered documents."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_redis.hmget.return_value = (None, None, None)
        aggregator._redis = mock_redis

        assert await aggregator.is_complete("unknown_doc") is False

    @pytest.mark.asyncio
    async def test_is_complete_returns_false_when_total_zero(self):
        """is_complete() returns False when total is 0 (edge case)."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_redis.hmget.return_value = ("0", "0", "processing")
        aggregator._redis = mock_redis

        assert await aggregator.is_complete("doc123") is False


class TestRegionAggregatorGetResult:
    """Tests for RegionAggregator.get_result()."""

    @pytest.mark.asyncio
    async def test_get_result_returns_all_region_results(self):
        """get_result() returns deserialized results for all regions."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        # Mock get_progress
        mock_redis.hmget.return_value = ("2", "2", "complete")
        # Mock hgetall
        mock_redis.hgetall.return_value = {
            "region_0": json.dumps({"text": "Hello"}),
            "region_1": json.dumps({"text": "World"}),
        }
        aggregator._redis = mock_redis

        result = await aggregator.get_result("doc123")

        assert result == {
            "region_0": {"text": "Hello"},
            "region_1": {"text": "World"},
        }

    @pytest.mark.asyncio
    async def test_get_result_raises_for_unregistered_document(self):
        """get_result() raises KeyError for unregistered documents."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_redis.hmget.return_value = (None, None, None)
        aggregator._redis = mock_redis

        with pytest.raises(KeyError, match="not registered"):
            await aggregator.get_result("unknown_doc")

    @pytest.mark.asyncio
    async def test_get_result_handles_malformed_json(self):
        """get_result() handles malformed JSON gracefully."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_redis.hmget.return_value = ("2", "2", "complete")
        mock_redis.hgetall.return_value = {
            "region_0": json.dumps({"text": "Good"}),
            "region_1": "not valid json {{{",
        }
        aggregator._redis = mock_redis

        result = await aggregator.get_result("doc123")

        # Good region should be deserialized
        assert result["region_0"] == {"text": "Good"}
        # Bad region should have raw payload and error
        assert "_raw" in result["region_1"]
        assert "_decode_error" in result["region_1"]

    @pytest.mark.asyncio
    async def test_get_result_applies_ttl_when_complete(self):
        """get_result() applies TTL to completed documents."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_redis.hmget.return_value = ("2", "2", "complete")
        mock_redis.hgetall.return_value = {
            "region_0": json.dumps({"text": "Hello"}),
            "region_1": json.dumps({"text": "World"}),
        }
        aggregator._redis = mock_redis

        await aggregator.get_result("doc123")

        # Verify TTL was applied to both keys
        expire_calls = mock_redis.expire.call_args_list
        assert len(expire_calls) == 2
        # Check that expire was called with correct keys and TTL
        keys_expired = [call[0][0] for call in expire_calls]
        assert "test:doc:doc123:meta" in keys_expired
        assert "test:doc:doc123:regions" in keys_expired

    @pytest.mark.asyncio
    async def test_get_result_does_not_apply_ttl_when_not_complete(self):
        """get_result() doesn't apply TTL when document is not complete."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="test",
        )

        mock_redis = AsyncMock()
        mock_redis.hmget.return_value = ("1", "3", "processing")
        mock_redis.hgetall.return_value = {
            "region_0": json.dumps({"text": "Hello"}),
        }
        aggregator._redis = mock_redis

        await aggregator.get_result("doc123")

        # Verify TTL was NOT applied
        mock_redis.expire.assert_not_called()


class TestRegionAggregatorKeyHelpers:
    """Tests for RegionAggregator key generation helpers."""

    def test_meta_key_format(self):
        """_meta_key() generates correct Redis key format."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="myapp",
        )

        assert aggregator._meta_key("doc123") == "myapp:doc:doc123:meta"

    def test_regions_key_format(self):
        """_regions_key() generates correct Redis key format."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="myapp",
        )

        assert aggregator._regions_key("doc123") == "myapp:doc:doc123:regions"

    def test_custom_key_prefix(self):
        """Custom key_prefix is used in generated keys."""
        from glmocr.aggregator import RegionAggregator

        aggregator = RegionAggregator(
            redis_url="redis://localhost:6379/0",
            key_prefix="custom_prefix",
        )

        assert aggregator._meta_key("test").startswith("custom_prefix:")
        assert aggregator._regions_key("test").startswith("custom_prefix:")
