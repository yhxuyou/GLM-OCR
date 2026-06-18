"""Comprehensive tests for async FastAPI server."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from glmocr.async_server import create_app


class MockConfig:
    """Mock GlmOcrConfig for testing."""

    def __init__(self):
        self.pipeline = MagicMock()
        self.pipeline.redis = MagicMock()
        self.pipeline.redis.url = "redis://localhost:6379/0"
        self.pipeline.redis.key_prefix = "test"
        self.pipeline.redis.max_connections = 10


@pytest.fixture
def mock_config():
    """Return a mock config instance."""
    return MockConfig()


@pytest.fixture
def mock_pipeline():
    """Return a mock Pipeline instance."""
    pipeline = MagicMock()
    pipeline.start = MagicMock()
    pipeline.stop = MagicMock()
    pipeline.process = MagicMock()
    return pipeline


@pytest.fixture
def mock_aggregator():
    """Return a mock RegionAggregator instance."""
    aggregator = AsyncMock()
    aggregator.connect = AsyncMock()
    aggregator.disconnect = AsyncMock()
    aggregator.register_document = AsyncMock()
    aggregator.get_progress = AsyncMock()
    aggregator.is_complete = AsyncMock()
    aggregator.get_result = AsyncMock()
    aggregator.on_region_complete = AsyncMock()
    return aggregator


@pytest.fixture
def app_with_mocks(mock_config, mock_pipeline, mock_aggregator):
    """Create FastAPI app with mocked dependencies."""
    with patch("glmocr.async_server.Pipeline") as MockPipeline, patch(
        "glmocr.async_server.RegionAggregator"
    ) as MockAggregator:

        MockPipeline.return_value = mock_pipeline
        MockAggregator.return_value = mock_aggregator

        app = create_app(mock_config)

        # Manually inject mocks into app.state
        app.state.pipeline = mock_pipeline
        app.state.aggregator = mock_aggregator

        yield app, mock_pipeline, mock_aggregator


@pytest.fixture
def client(app_with_mocks):
    """Return TestClient for the app."""
    app, _, _ = app_with_mocks
    return TestClient(app)


class TestHealthCheck:
    """Tests for GET /health endpoint."""

    def test_health_check_returns_ok(self, client):
        """Health check endpoint returns status ok."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestParseAsync:
    """Tests for POST /parse/async endpoint."""

    def test_parse_async_successful_upload(self, client, mock_aggregator):
        """POST /parse/async accepts file and returns doc_id."""
        mock_aggregator.register_document.return_value = None

        file_content = b"fake image content"
        files = {"file": ("test.png", file_content, "image/png")}

        response = client.post("/parse/async", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "doc_id" in data
        assert data["status"] == "processing"
        assert len(data["doc_id"]) == 36  # UUID format

        # Verify aggregator was called
        mock_aggregator.register_document.assert_called_once()
        call_args = mock_aggregator.register_document.call_args
        assert call_args[1]["total_regions"] == 1

    def test_parse_async_empty_file(self, client):
        """POST /parse/async rejects empty file."""
        files = {"file": ("empty.png", b"", "image/png")}

        response = client.post("/parse/async", files=files)

        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    def test_parse_async_no_file(self, client):
        """POST /parse/async rejects request without file."""
        response = client.post("/parse/async")

        assert response.status_code == 422  # Validation error

    def test_parse_async_pdf_upload(self, client, mock_aggregator):
        """POST /parse/async accepts PDF files."""
        mock_aggregator.register_document.return_value = None

        file_content = b"%PDF-1.4 fake pdf content"
        files = {"file": ("document.pdf", file_content, "application/pdf")}

        response = client.post("/parse/async", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"

    def test_parse_async_jpg_upload(self, client, mock_aggregator):
        """POST /parse/async accepts JPG images."""
        mock_aggregator.register_document.return_value = None

        file_content = b"\xff\xd8\xff\xe0 fake jpeg"
        files = {"file": ("photo.jpg", file_content, "image/jpeg")}

        response = client.post("/parse/async", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "doc_id" in data

    def test_parse_async_aggregator_error(self, client, mock_aggregator):
        """POST /parse/async handles aggregator registration failure."""
        mock_aggregator.register_document.side_effect = Exception("Redis connection failed")

        file_content = b"test content"
        files = {"file": ("test.png", file_content, "image/png")}

        response = client.post("/parse/async", files=files)

        assert response.status_code == 500
        assert "Failed to submit document" in response.json()["detail"]


class TestGetStatus:
    """Tests for GET /parse/status/{doc_id} endpoint."""

    def test_get_status_processing(self, client, mock_aggregator):
        """GET /parse/status/{doc_id} returns processing status."""
        mock_aggregator.get_progress.return_value = {
            "completed": 2,
            "total": 5,
            "status": "processing",
        }

        response = client.get("/parse/status/test-doc-123")

        assert response.status_code == 200
        data = response.json()
        assert data["completed"] == 2
        assert data["total"] == 5
        assert data["status"] == "processing"

    def test_get_status_completed(self, client, mock_aggregator):
        """GET /parse/status/{doc_id} returns completed status."""
        mock_aggregator.get_progress.return_value = {
            "completed": 5,
            "total": 5,
            "status": "complete",
        }

        response = client.get("/parse/status/test-doc-123")

        assert response.status_code == 200
        data = response.json()
        assert data["completed"] == 5
        assert data["total"] == 5
        assert data["status"] == "completed"

    def test_get_status_not_found(self, client, mock_aggregator):
        """GET /parse/status/{doc_id} returns not_found for unknown doc."""
        mock_aggregator.get_progress.return_value = {
            "completed": 0,
            "total": 0,
            "status": "unknown",
        }

        response = client.get("/parse/status/unknown-doc")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_found"
        assert data["completed"] == 0
        assert data["total"] == 0

    def test_get_status_aggregator_error(self, client, mock_aggregator):
        """GET /parse/status/{doc_id} handles aggregator failure."""
        mock_aggregator.get_progress.side_effect = Exception("Redis error")

        response = client.get("/parse/status/test-doc-123")

        assert response.status_code == 500
        assert "Failed to get status" in response.json()["detail"]


class TestGetResult:
    """Tests for GET /parse/result/{doc_id} endpoint."""

    def test_get_result_success(self, client, mock_aggregator):
        """GET /parse/result/{doc_id} returns result when complete."""
        mock_aggregator.get_progress.return_value = {
            "completed": 5,
            "total": 5,
            "status": "complete",
        }
        mock_aggregator.is_complete.return_value = True
        mock_aggregator.get_result.return_value = {
            "region_0": {"text": "Hello", "bbox": [10, 20, 100, 200]},
            "region_1": {"text": "World", "bbox": [10, 220, 100, 400]},
        }

        response = client.get("/parse/result/test-doc-123")

        assert response.status_code == 200
        data = response.json()
        assert "region_0" in data
        assert "region_1" in data
        assert data["region_0"]["text"] == "Hello"

    def test_get_result_not_found(self, client, mock_aggregator):
        """GET /parse/result/{doc_id} returns 404 for unknown doc."""
        mock_aggregator.get_progress.return_value = {
            "completed": 0,
            "total": 0,
            "status": "unknown",
        }

        response = client.get("/parse/result/unknown-doc")

        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_get_result_not_complete(self, client, mock_aggregator):
        """GET /parse/result/{doc_id} returns 202 when still processing."""
        mock_aggregator.get_progress.return_value = {
            "completed": 2,
            "total": 5,
            "status": "processing",
        }
        mock_aggregator.is_complete.return_value = False

        response = client.get("/parse/result/test-doc-123")

        assert response.status_code == 202
        assert "not yet complete" in response.json()["detail"]

    def test_get_result_aggregator_error(self, client, mock_aggregator):
        """GET /parse/result/{doc_id} handles aggregator failure."""
        mock_aggregator.get_progress.side_effect = Exception("Redis error")

        response = client.get("/parse/result/test-doc-123")

        assert response.status_code == 500
        assert "Failed to get result" in response.json()["detail"]


class TestWebSocket:
    """Tests for WebSocket /ws/{doc_id} endpoint."""

    def test_websocket_not_found(self, client, mock_aggregator):
        """WebSocket /ws/{doc_id} closes with error for unknown doc."""
        mock_aggregator.get_progress.return_value = {
            "completed": 0,
            "total": 0,
            "status": "unknown",
        }

        with client.websocket_connect("/ws/unknown-doc") as websocket:
            # Should receive error message
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "Document not found" in data["message"]

    def test_websocket_progress_updates(self, client, mock_aggregator):
        """WebSocket /ws/{doc_id} sends progress updates."""
        # The websocket handler calls get_progress twice before the loop:
        # once for the "document exists" check, then again at the top of the while loop.
        mock_aggregator.get_progress.side_effect = [
            {"completed": 0, "total": 3, "status": "processing"},  # initial existence check
            {"completed": 1, "total": 3, "status": "processing"},  # first loop iteration
            {"completed": 2, "total": 3, "status": "processing"},  # second loop iteration
            {"completed": 3, "total": 3, "status": "complete"},    # third loop iteration
        ]
        mock_aggregator.get_result.return_value = {
            "region_0": {"text": "First"},
            "region_1": {"text": "Second"},
            "region_2": {"text": "Third"},
        }

        with client.websocket_connect("/ws/test-doc-123") as websocket:
            # Receive first progress
            data = websocket.receive_json()
            assert data["type"] == "progress"
            assert data["completed"] == 1
            assert data["total"] == 3

            # Receive second progress
            data = websocket.receive_json()
            assert data["type"] == "progress"
            assert data["completed"] == 2
            assert data["total"] == 3

            # Receive final progress and completion
            data = websocket.receive_json()
            assert data["type"] == "progress"
            assert data["completed"] == 3
            assert data["total"] == 3

            # Receive completion message with result
            data = websocket.receive_json()
            assert data["type"] == "complete"
            assert "result" in data
            assert len(data["result"]) == 3

    def test_websocket_already_complete(self, client, mock_aggregator):
        """WebSocket /ws/{doc_id} immediately completes if doc already done."""
        mock_aggregator.get_progress.return_value = {
            "completed": 5,
            "total": 5,
            "status": "complete",
        }
        mock_aggregator.get_result.return_value = {
            "region_0": {"text": "Complete result"},
        }

        with client.websocket_connect("/ws/test-doc-123") as websocket:
            # Should receive progress then complete immediately
            data = websocket.receive_json()
            assert data["type"] == "progress"
            assert data["completed"] == 5

            data = websocket.receive_json()
            assert data["type"] == "complete"
            assert "result" in data


class TestAppLifecycle:
    """Tests for application startup and shutdown."""

    def test_startup_initializes_pipeline_and_aggregator(self, mock_config):
        """Startup event initializes Pipeline and RegionAggregator."""
        with patch("glmocr.async_server.Pipeline") as MockPipeline, patch(
            "glmocr.async_server.RegionAggregator"
        ) as MockAggregator:

            mock_pipeline = MagicMock()
            mock_aggregator = AsyncMock()
            MockPipeline.return_value = mock_pipeline
            MockAggregator.return_value = mock_aggregator

            app = create_app(mock_config)

            # Trigger startup
            with TestClient(app) as client:
                # Verify pipeline was started
                mock_pipeline.start.assert_called_once()
                # Verify aggregator was connected
                mock_aggregator.connect.assert_called_once()

    def test_shutdown_stops_pipeline_and_disconnects_aggregator(self, mock_config):
        """Shutdown event stops Pipeline and disconnects RegionAggregator."""
        with patch("glmocr.async_server.Pipeline") as MockPipeline, patch(
            "glmocr.async_server.RegionAggregator"
        ) as MockAggregator:

            mock_pipeline = MagicMock()
            mock_aggregator = AsyncMock()
            MockPipeline.return_value = mock_pipeline
            MockAggregator.return_value = mock_aggregator

            app = create_app(mock_config)

            # Trigger startup and shutdown
            with TestClient(app) as client:
                pass  # Client context triggers startup/shutdown

            # Verify cleanup
            mock_pipeline.stop.assert_called_once()
            mock_aggregator.disconnect.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_async_large_file(self, client, mock_aggregator):
        """POST /parse/async handles large file uploads."""
        mock_aggregator.register_document.return_value = None

        # Create a 1MB file
        large_content = b"x" * (1024 * 1024)
        files = {"file": ("large.png", large_content, "image/png")}

        response = client.post("/parse/async", files=files)

        assert response.status_code == 200
        assert "doc_id" in response.json()

    def test_parse_async_special_characters_in_filename(self, client, mock_aggregator):
        """POST /parse/async handles filenames with special characters."""
        mock_aggregator.register_document.return_value = None

        file_content = b"test content"
        files = {"file": ("test-file_v1.0 (copy).png", file_content, "image/png")}

        response = client.post("/parse/async", files=files)

        assert response.status_code == 200

    def test_get_status_with_uuid_doc_id(self, client, mock_aggregator):
        """GET /parse/status/{doc_id} works with UUID format doc_id."""
        mock_aggregator.get_progress.return_value = {
            "completed": 0,
            "total": 1,
            "status": "processing",
        }

        uuid_doc_id = "123e4567-e89b-12d3-a456-426614174000"
        response = client.get(f"/parse/status/{uuid_doc_id}")

        assert response.status_code == 200

    def test_get_result_empty_result(self, client, mock_aggregator):
        """GET /parse/result/{doc_id} handles empty result."""
        mock_aggregator.get_progress.return_value = {
            "completed": 1,
            "total": 1,
            "status": "complete",
        }
        mock_aggregator.is_complete.return_value = True
        mock_aggregator.get_result.return_value = {}

        response = client.get("/parse/result/test-doc-123")

        assert response.status_code == 200
        assert response.json() == {}

    def test_websocket_client_disconnect_during_processing(self, client, mock_aggregator):
        """WebSocket handles client disconnect gracefully."""
        # Simulate long processing
        mock_aggregator.get_progress.side_effect = [
            {"completed": 1, "total": 10, "status": "processing"},
            {"completed": 2, "total": 10, "status": "processing"},
        ]

        with client.websocket_connect("/ws/test-doc-123") as websocket:
            # Receive one progress update
            data = websocket.receive_json()
            assert data["type"] == "progress"
            # Client disconnects - should not raise error on server

    def test_multiple_concurrent_uploads(self, client, mock_aggregator):
        """Multiple concurrent uploads get unique doc_ids."""
        mock_aggregator.register_document.return_value = None

        doc_ids = []
        for i in range(5):
            file_content = f"content {i}".encode()
            files = {"file": (f"test{i}.png", file_content, "image/png")}
            response = client.post("/parse/async", files=files)
            assert response.status_code == 200
            doc_ids.append(response.json()["doc_id"])

        # All doc_ids should be unique
        assert len(set(doc_ids)) == 5


class TestBackgroundProcessing:
    """Tests for background document processing."""

    @pytest.mark.asyncio
    async def test_background_process_updates_aggregator(self, mock_aggregator):
        """Background task calls aggregator.on_region_complete for each result."""
        from glmocr.async_server import _process_document_background

        # Mock pipeline to return results
        mock_result = MagicMock()
        mock_result.json_result = {"text": "OCR text"}
        mock_result.markdown_result = "# OCR text"
        mock_result.original_images = ["base64_image"]

        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = [mock_result]

        await _process_document_background(
            doc_id="test-doc",
            file_content=b"test",
            filename="test.png",
            pipeline=mock_pipeline,
            aggregator=mock_aggregator,
        )

        # Verify aggregator was updated
        mock_aggregator.on_region_complete.assert_called_once()
        call_args = mock_aggregator.on_region_complete.call_args
        assert call_args[0][0] == "test-doc"  # doc_id
        assert call_args[0][1] == "unit_0"  # region_id

    @pytest.mark.asyncio
    async def test_background_process_handles_empty_results(self, mock_aggregator):
        """Background task handles when pipeline produces no results."""
        from glmocr.async_server import _process_document_background

        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = []

        await _process_document_background(
            doc_id="test-doc",
            file_content=b"test",
            filename="test.png",
            pipeline=mock_pipeline,
            aggregator=mock_aggregator,
        )

        # Should mark as complete with empty result
        mock_aggregator.on_region_complete.assert_called_once()
        call_args = mock_aggregator.on_region_complete.call_args
        assert call_args[0][1] == "empty"

    @pytest.mark.asyncio
    async def test_background_process_handles_pipeline_error(self, mock_aggregator):
        """Background task handles pipeline errors gracefully."""
        from glmocr.async_server import _process_document_background

        mock_pipeline = MagicMock()
        mock_pipeline.process.side_effect = Exception("Pipeline failed")

        # Should not raise
        await _process_document_background(
            doc_id="test-doc",
            file_content=b"test",
            filename="test.png",
            pipeline=mock_pipeline,
            aggregator=mock_aggregator,
        )

    @pytest.mark.asyncio
    async def test_background_process_multiple_regions(self, mock_aggregator):
        """Background task processes multiple regions from pipeline."""
        from glmocr.async_server import _process_document_background

        # Create multiple mock results
        mock_results = []
        for i in range(3):
            result = MagicMock()
            result.json_result = {"text": f"Region {i}"}
            result.markdown_result = f"# Region {i}"
            result.original_images = [f"image_{i}"]
            mock_results.append(result)

        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = mock_results

        await _process_document_background(
            doc_id="test-doc",
            file_content=b"test",
            filename="test.png",
            pipeline=mock_pipeline,
            aggregator=mock_aggregator,
        )

        # Should call aggregator for each region
        assert mock_aggregator.on_region_complete.call_count == 3

        # Verify region IDs
        calls = mock_aggregator.on_region_complete.call_args_list
        region_ids = [call[0][1] for call in calls]
        assert region_ids == ["unit_0", "unit_1", "unit_2"]
