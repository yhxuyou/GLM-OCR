"""Tests for async components: AsyncOCRClient, AsyncPipeline, and async configs."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock

import pytest
import httpx

from glmocr.async_ocr_client import AsyncOCRClient
from glmocr.async_pipeline import AsyncPipeline
from glmocr.config import AsyncOCRApiConfig, AsyncPipelineConfig, PipelineConfig


# ============================================================================
# AsyncOCRClient Tests
# ============================================================================


class TestAsyncOCRClient:
    """Tests for AsyncOCRClient."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock OCRApiConfig for testing."""
        config = MagicMock()
        config.api_host = "localhost"
        config.api_port = 5002
        config.api_scheme = "http"
        config.api_path = "/v1/chat/completions"
        config.api_url = None
        config.api_key = "test-api-key"
        config.headers = {"X-Custom": "header"}
        config.verify_ssl = False
        config.connect_timeout = 30
        config.request_timeout = 120
        config.model = "test-model"
        config.api_mode = "openai"
        config.retry_max_attempts = 2
        config.retry_backoff_base_seconds = 0.5
        config.retry_backoff_max_seconds = 8.0
        config.retry_jitter_ratio = 0.2
        config.retry_status_codes = [429, 500, 502, 503, 504]
        config.connection_pool_size = 128
        return config

    def test_async_client_init(self, mock_config):
        """Test AsyncOCRClient initialization."""
        client = AsyncOCRClient(mock_config)

        assert client.api_host == "localhost"
        assert client.api_port == 5002
        assert client.api_scheme == "http"
        assert client.api_path == "/v1/chat/completions"
        assert client.api_url == "http://localhost:5002/v1/chat/completions"
        assert client.api_key == "test-api-key"
        assert client.extra_headers == {"X-Custom": "header"}
        assert client.verify_ssl is False
        assert client.connect_timeout == 30
        assert client.request_timeout == 120
        assert client.retry_max_attempts == 2
        assert client.retry_backoff_base_seconds == 0.5
        assert client.retry_backoff_max_seconds == 8.0
        assert client.retry_jitter_ratio == 0.2
        assert client.retry_status_codes == {429, 500, 502, 503, 504}
        assert client._pool_maxsize == 128
        assert client.model == "test-model"
        assert client.api_mode == "openai"
        assert client._client is None

    @pytest.mark.asyncio
    async def test_async_client_start_stop(self, mock_config):
        """Test AsyncOCRClient start and stop lifecycle."""
        client = AsyncOCRClient(mock_config)

        # Initially, client should be None
        assert client._client is None

        # Start the client
        await client.start()
        assert client._client is not None
        assert isinstance(client._client, httpx.AsyncClient)

        # Verify connection pool settings
        assert client._client._transport._pool._max_connections == 128
        assert client._client._transport._pool._max_keepalive_connections == 128

        # Stop the client
        await client.stop()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_async_process_single_request(self, mock_config):
        """Test processing a single async request with mocked httpx."""
        client = AsyncOCRClient(mock_config)
        await client.start()

        # Mock the httpx.AsyncClient.post method
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OCR result text"}}]
        }

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            request_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Test"}],
                    }
                ]
            }

            result, status_code = await client.async_process(request_data)

            assert status_code == 200
            assert result["choices"][0]["message"]["content"] == "OCR result text"
            mock_post.assert_called_once()

        await client.stop()

    @pytest.mark.asyncio
    async def test_async_process_batch(self, mock_config):
        """Test batch processing of multiple async requests."""
        client = AsyncOCRClient(mock_config)
        await client.start()

        # Mock the httpx.AsyncClient.post method
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Result"}}]
        }

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            requests = [
                {"messages": [{"role": "user", "content": [{"type": "text", "text": "Req1"}]}]},
                {"messages": [{"role": "user", "content": [{"type": "text", "text": "Req2"}]}]},
                {"messages": [{"role": "user", "content": [{"type": "text", "text": "Req3"}]}]},
            ]

            results = await client.async_process_batch(requests)

            assert len(results) == 3
            assert all(status == 200 for _, status in results)
            assert mock_post.call_count == 3

        await client.stop()

    @pytest.mark.asyncio
    async def test_async_retry_logic(self, mock_config):
        """Test retry logic with transient failures."""
        client = AsyncOCRClient(mock_config)
        await client.start()

        # Mock responses: first two fail with 503, third succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 503
        mock_response_fail.text = "Service Unavailable"
        mock_response_fail.headers = {}

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "choices": [{"message": {"content": "Success"}}]
        }

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            # First two calls fail, third succeeds
            mock_post.side_effect = [
                mock_response_fail,
                mock_response_fail,
                mock_response_success,
            ]

            # Mock sleep to avoid actual delays
            with patch("asyncio.sleep", new_callable=AsyncMock):
                request_data = {
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Test"}]}]
                }

                result, status_code = await client.async_process(request_data)

                assert status_code == 200
                assert result["choices"][0]["message"]["content"] == "Success"
                assert mock_post.call_count == 3

        await client.stop()

    def test_async_connection_pool(self, mock_config):
        """Test connection pool configuration."""
        # Test default pool size
        client1 = AsyncOCRClient(mock_config)
        assert client1._pool_maxsize == 128

        # Test custom pool size
        mock_config.connection_pool_size = 256
        client2 = AsyncOCRClient(mock_config)
        assert client2._pool_maxsize == 256

        # Test None pool size (should default to 128)
        mock_config.connection_pool_size = None
        client3 = AsyncOCRClient(mock_config)
        assert client3._pool_maxsize == 128


# ============================================================================
# AsyncPipeline Tests
# ============================================================================


class TestAsyncPipeline:
    """Tests for AsyncPipeline."""

    @pytest.fixture
    def mock_pipeline_config(self):
        """Create a mock PipelineConfig for testing."""
        config = MagicMock(spec=PipelineConfig)
        config.page_loader = MagicMock()
        config.ocr_api = MagicMock()
        config.ocr_api.api_host = "localhost"
        config.ocr_api.api_port = 5002
        config.ocr_api.api_scheme = "http"
        config.ocr_api.api_path = "/v1/chat/completions"
        config.ocr_api.api_url = None
        config.ocr_api.api_key = "test-key"
        config.ocr_api.headers = {}
        config.ocr_api.verify_ssl = False
        config.ocr_api.connect_timeout = 30
        config.ocr_api.request_timeout = 120
        config.ocr_api.model = "test-model"
        config.ocr_api.api_mode = "openai"
        config.ocr_api.retry_max_attempts = 2
        config.ocr_api.retry_backoff_base_seconds = 0.5
        config.ocr_api.retry_backoff_max_seconds = 8.0
        config.ocr_api.retry_jitter_ratio = 0.2
        config.ocr_api.retry_status_codes = [429, 500, 502, 503, 504]
        config.ocr_api.connection_pool_size = 128
        config.result_formatter = MagicMock()
        config.layout = MagicMock()
        config.layout.use_polygon = False
        config.max_concurrent_regions = 100
        return config

    def test_async_pipeline_init(self, mock_pipeline_config):
        """Test AsyncPipeline initialization."""
        with patch("glmocr.async_pipeline.PageLoader"), \
             patch("glmocr.async_pipeline.AsyncOCRClient"), \
             patch("glmocr.async_pipeline.ResultFormatter"), \
             patch("glmocr.layout.PPDocLayoutDetector"):

            pipeline = AsyncPipeline(mock_pipeline_config)

            assert pipeline.config == mock_pipeline_config
            assert pipeline.page_loader is not None
            assert pipeline.async_ocr_client is not None
            assert pipeline.result_formatter is not None
            assert pipeline.layout_detector is not None
            assert isinstance(pipeline._region_semaphore, asyncio.Semaphore)

    @pytest.mark.asyncio
    async def test_async_pipeline_start_stop(self, mock_pipeline_config):
        """Test AsyncPipeline start and stop lifecycle."""
        with patch("glmocr.async_pipeline.PageLoader"), \
             patch("glmocr.async_pipeline.AsyncOCRClient") as MockOCRClient, \
             patch("glmocr.async_pipeline.ResultFormatter"), \
             patch("glmocr.layout.PPDocLayoutDetector") as MockLayoutDetector:

            mock_ocr_client = AsyncMock()
            mock_ocr_client.start = AsyncMock()
            mock_ocr_client.stop = AsyncMock()
            MockOCRClient.return_value = mock_ocr_client

            mock_layout_detector = MagicMock()
            mock_layout_detector.start = MagicMock()
            mock_layout_detector.stop = MagicMock()
            MockLayoutDetector.return_value = mock_layout_detector

            pipeline = AsyncPipeline(mock_pipeline_config)

            # Start the pipeline
            await pipeline.start()
            mock_layout_detector.start.assert_called_once()
            mock_ocr_client.start.assert_called_once()

            # Stop the pipeline
            await pipeline.stop()
            mock_ocr_client.stop.assert_called_once()
            mock_layout_detector.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_process_async(self, mock_pipeline_config):
        """Test async processing flow with mocked dependencies."""
        with patch("glmocr.async_pipeline.PageLoader") as MockPageLoader, \
             patch("glmocr.async_pipeline.AsyncOCRClient") as MockOCRClient, \
             patch("glmocr.async_pipeline.ResultFormatter"), \
             patch("glmocr.layout.PPDocLayoutDetector") as MockLayoutDetector:

            # Setup mocks
            mock_page_loader = MagicMock()
            MockPageLoader.return_value = mock_page_loader

            mock_ocr_client = AsyncMock()
            MockOCRClient.return_value = mock_ocr_client

            mock_layout_detector = MagicMock()
            MockLayoutDetector.return_value = mock_layout_detector

            pipeline = AsyncPipeline(mock_pipeline_config)

            # Mock internal methods
            mock_aggregator = AsyncMock()
            mock_aggregator.register_document = AsyncMock()
            mock_aggregator.on_region_complete = AsyncMock()

            # Mock extract_image_sources to return empty (passthrough case)
            with patch("glmocr.async_pipeline.extract_image_sources", return_value=[]):
                # Mock _process_passthrough_async
                mock_result = MagicMock()
                mock_result.json_result = {"text": "result"}
                mock_result.markdown_result = "# result"

                with patch.object(
                    pipeline, "_process_passthrough_async", new_callable=AsyncMock
                ) as mock_passthrough:
                    mock_passthrough.return_value = mock_result

                    request_data = {"messages": []}
                    doc_id = await pipeline.process_async(
                        request_data, "test-doc-123", mock_aggregator
                    )

                    assert doc_id == "test-doc-123"
                    mock_aggregator.register_document.assert_called_once_with(
                        "test-doc-123", total_regions=1
                    )
                    mock_aggregator.on_region_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_semaphore_control(self, mock_pipeline_config):
        """Test semaphore-based concurrency control for regions."""
        with patch("glmocr.async_pipeline.PageLoader"), \
             patch("glmocr.async_pipeline.AsyncOCRClient"), \
             patch("glmocr.async_pipeline.ResultFormatter"), \
             patch("glmocr.layout.PPDocLayoutDetector"):

            # Set custom concurrency limit
            mock_pipeline_config.max_concurrent_regions = 5
            pipeline = AsyncPipeline(mock_pipeline_config)

            # Verify semaphore is initialized with correct value
            assert isinstance(pipeline._region_semaphore, asyncio.Semaphore)
            # The semaphore should allow up to 5 concurrent operations
            # We can't directly check the value, but we can verify it exists

    @pytest.mark.asyncio
    async def test_async_region_submission(self, mock_pipeline_config):
        """Test asynchronous region submission with OCR calls."""
        with patch("glmocr.async_pipeline.PageLoader") as MockPageLoader, \
             patch("glmocr.async_pipeline.AsyncOCRClient") as MockOCRClient, \
             patch("glmocr.async_pipeline.ResultFormatter"), \
             patch("glmocr.layout.PPDocLayoutDetector") as MockLayoutDetector:

            # Setup mocks
            mock_page_loader = MagicMock()
            MockPageLoader.return_value = mock_page_loader

            mock_ocr_client = AsyncMock()
            mock_response = {"choices": [{"message": {"content": "OCR text"}}]}
            mock_ocr_client.async_process = AsyncMock(
                return_value=(mock_response, 200)
            )
            MockOCRClient.return_value = mock_ocr_client

            mock_layout_detector = MagicMock()
            MockLayoutDetector.return_value = mock_layout_detector

            pipeline = AsyncPipeline(mock_pipeline_config)

            # Mock aggregator
            mock_aggregator = AsyncMock()
            mock_aggregator.on_region_complete = AsyncMock()

            # Mock image cropping
            mock_page = MagicMock()
            region_info = {
                "page_idx": 0,
                "page": mock_page,
                "region": {
                    "bbox_2d": [10, 10, 100, 100],
                    "task_type": "text",
                },
            }

            # Mock crop_image_region
            with patch("glmocr.utils.image_utils.crop_image_region") as mock_crop:
                mock_cropped = MagicMock()
                mock_crop.return_value = mock_cropped

                # Mock build_request_from_image
                mock_page_loader.build_request_from_image.return_value = {
                    "messages": []
                }

                await pipeline._submit_region_async(
                    "test-doc", "region_0", region_info, mock_aggregator
                )

                # Verify OCR was called
                mock_ocr_client.async_process.assert_called_once()

                # Verify aggregator was notified
                mock_aggregator.on_region_complete.assert_called_once()
                call_args = mock_aggregator.on_region_complete.call_args
                assert call_args[0][0] == "test-doc"
                assert call_args[0][1] == "region_0"
                result = call_args[0][2]
                assert result["content"] == "OCR text"


# ============================================================================
# Configuration Tests
# ============================================================================


class TestAsyncConfigs:
    """Tests for async configuration classes."""

    def test_async_ocr_config(self):
        """Test AsyncOCRApiConfig initialization and defaults."""
        config = AsyncOCRApiConfig()

        # Check inherited fields from OCRApiConfig
        assert config.api_host == "localhost"
        assert config.api_port == 5002
        assert config.api_scheme is None
        assert config.api_path == "/v1/chat/completions"
        assert config.api_url is None
        assert config.api_key is None
        assert config.model is None
        assert config.headers == {}
        assert config.verify_ssl is False
        assert config.api_mode == "openai"
        assert config.connect_timeout == 30
        assert config.request_timeout == 120
        assert config.retry_max_attempts == 2
        assert config.retry_backoff_base_seconds == 0.5
        assert config.retry_backoff_max_seconds == 8.0
        assert config.retry_jitter_ratio == 0.2
        assert config.retry_status_codes == [429, 500, 502, 503, 504]
        assert config.connection_pool_size == 128

        # Check AsyncOCRApiConfig-specific fields
        assert config.max_connections == 100
        assert config.max_concurrent_requests == 50

        # Test custom values
        custom_config = AsyncOCRApiConfig(
            api_host="custom-host",
            api_port=8080,
            max_connections=200,
            max_concurrent_requests=100,
        )
        assert custom_config.api_host == "custom-host"
        assert custom_config.api_port == 8080
        assert custom_config.max_connections == 200
        assert custom_config.max_concurrent_requests == 100

    def test_async_pipeline_config(self):
        """Test AsyncPipelineConfig initialization and defaults."""
        config = AsyncPipelineConfig()

        # Check default values
        assert config.max_concurrent_regions == 100
        assert config.enable_batch_processing is True

        # Test custom values
        custom_config = AsyncPipelineConfig(
            max_concurrent_regions=50,
            enable_batch_processing=False,
        )
        assert custom_config.max_concurrent_regions == 50
        assert custom_config.enable_batch_processing is False

    def test_config_env_vars(self):
        """Test environment variable mapping for async configs."""
        from glmocr.config import _ENV_MAP

        # Check that async-related env vars are mapped
        assert "ASYNC_OCR_MAX_CONNECTIONS" in _ENV_MAP
        assert _ENV_MAP["ASYNC_OCR_MAX_CONNECTIONS"] == "pipeline.async_ocr.max_connections"

        assert "ASYNC_OCR_MAX_CONCURRENT_REQUESTS" in _ENV_MAP
        assert (
            _ENV_MAP["ASYNC_OCR_MAX_CONCURRENT_REQUESTS"]
            == "pipeline.async_ocr.max_concurrent_requests"
        )

        assert "ASYNC_PIPELINE_MAX_CONCURRENT_REGIONS" in _ENV_MAP
        assert (
            _ENV_MAP["ASYNC_PIPELINE_MAX_CONCURRENT_REGIONS"]
            == "pipeline.async_pipeline.max_concurrent_regions"
        )

        # Test that PipelineConfig includes async configs
        pipeline_config = PipelineConfig()
        assert hasattr(pipeline_config, "async_ocr")
        assert hasattr(pipeline_config, "async_pipeline")
        assert isinstance(pipeline_config.async_ocr, AsyncOCRApiConfig)
        assert isinstance(pipeline_config.async_pipeline, AsyncPipelineConfig)
