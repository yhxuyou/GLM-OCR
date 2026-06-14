"""Integration tests for Pipeline.process_async() and async region submission."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestProcessAsync:
    """Tests for Pipeline.process_async() method."""

    @pytest.fixture
    def mock_pipeline_config(self):
        """Create a minimal PipelineConfig for testing."""
        from glmocr.config import PipelineConfig

        config = PipelineConfig()
        config.layout.use_polygon = False
        return config

    @pytest.fixture
    def mock_aggregator(self):
        """Create a mock RegionAggregator."""
        aggregator = AsyncMock()
        aggregator.register_document = AsyncMock()
        aggregator.on_region_complete = AsyncMock()
        return aggregator

    @pytest.mark.asyncio
    async def test_process_async_no_images_processes_passthrough(
        self, mock_pipeline_config, mock_aggregator
    ):
        """process_async() with no image sources processes as passthrough."""
        from glmocr.pipeline import Pipeline
        from glmocr.parser_result import PipelineResult

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config
            pipeline.page_loader = MagicMock()
            pipeline.ocr_client = MagicMock()
            pipeline.result_formatter = MagicMock()
            pipeline.layout_detector = MagicMock()

            # Mock _process_passthrough to return a result
            mock_result = PipelineResult(
                json_result=[{"text": "hello"}],
                markdown_result="# Hello",
                original_images=[],
            )
            pipeline._process_passthrough = MagicMock(return_value=mock_result)

            request_data = {"messages": [{"role": "user", "content": []}]}
            doc_id = await pipeline.process_async(
                request_data, "doc123", mock_aggregator
            )

            assert doc_id == "doc123"
            mock_aggregator.register_document.assert_called_once_with(
                "doc123", total_regions=1
            )
            mock_aggregator.on_region_complete.assert_called_once()
            call_args = mock_aggregator.on_region_complete.call_args
            assert call_args[0][0] == "doc123"
            assert call_args[0][1] == "passthrough"

    @pytest.mark.asyncio
    async def test_process_async_empty_pages_registers_empty_result(
        self, mock_pipeline_config, mock_aggregator
    ):
        """process_async() with no pages registers an empty result."""
        from glmocr.pipeline import Pipeline

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config
            pipeline.page_loader = MagicMock()
            pipeline.ocr_client = MagicMock()
            pipeline.layout_detector = MagicMock()

            # Mock page loader to return no pages
            pipeline.page_loader.iter_pages_with_unit_indices = MagicMock(
                return_value=iter([])
            )

            request_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "file:///test.png"},
                            }
                        ],
                    }
                ]
            }
            doc_id = await pipeline.process_async(
                request_data, "doc456", mock_aggregator
            )

            assert doc_id == "doc456"
            mock_aggregator.register_document.assert_called_once_with(
                "doc456", total_regions=1
            )
            mock_aggregator.on_region_complete.assert_called_once()
            call_args = mock_aggregator.on_region_complete.call_args
            assert call_args[0][1] == "empty"

    @pytest.mark.asyncio
    async def test_process_async_with_regions_registers_correct_count(
        self, mock_pipeline_config, mock_aggregator
    ):
        """process_async() registers document with correct region count."""
        from glmocr.pipeline import Pipeline
        from PIL import Image

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config
            pipeline.page_loader = MagicMock()
            pipeline.ocr_client = MagicMock()
            pipeline.layout_detector = MagicMock()

            # Mock page loader to return 2 pages
            mock_page1 = Image.new("RGB", (100, 100))
            mock_page2 = Image.new("RGB", (100, 100))
            pipeline.page_loader.iter_pages_with_unit_indices = MagicMock(
                return_value=iter([(mock_page1, 0), (mock_page2, 1)])
            )

            # Mock layout detector to return 3 regions per page
            mock_regions = [
                {"bbox_2d": [10, 10, 50, 50], "label": "text"},
                {"bbox_2d": [60, 10, 90, 50], "label": "text"},
                {"bbox_2d": [10, 60, 90, 90], "label": "image"},
            ]
            pipeline.layout_detector.process = MagicMock(
                return_value=([mock_regions], [])
            )

            request_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "file:///test.png"},
                            }
                        ],
                    }
                ]
            }

            # Patch _submit_region_async to avoid actual OCR calls
            with patch.object(
                pipeline, "_submit_region_async", new_callable=AsyncMock
            ):
                doc_id = await pipeline.process_async(
                    request_data, "doc789", mock_aggregator
                )

            assert doc_id == "doc789"
            # 2 pages × 3 regions = 6 total regions
            mock_aggregator.register_document.assert_called_once_with(
                "doc789", total_regions=6
            )

    @pytest.mark.asyncio
    async def test_process_async_layout_failure_skips_page(
        self, mock_pipeline_config, mock_aggregator
    ):
        """process_async() skips pages where layout detection fails."""
        from glmocr.pipeline import Pipeline
        from PIL import Image

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config
            pipeline.page_loader = MagicMock()
            pipeline.ocr_client = MagicMock()
            pipeline.layout_detector = MagicMock()

            mock_page = Image.new("RGB", (100, 100))
            pipeline.page_loader.iter_pages_with_unit_indices = MagicMock(
                return_value=iter([(mock_page, 0)])
            )

            # Layout detector raises an exception
            pipeline.layout_detector.process = MagicMock(
                side_effect=RuntimeError("Layout model crashed")
            )

            request_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "file:///test.png"},
                            }
                        ],
                    }
                ]
            }

            with patch.object(
                pipeline, "_submit_region_async", new_callable=AsyncMock
            ):
                doc_id = await pipeline.process_async(
                    request_data, "doc_fail", mock_aggregator
                )

            assert doc_id == "doc_fail"
            # No regions detected, but should still register with total_regions=1
            mock_aggregator.register_document.assert_called_once_with(
                "doc_fail", total_regions=1
            )
            # Should call on_region_complete with "no_regions"
            mock_aggregator.on_region_complete.assert_called_once()
            call_args = mock_aggregator.on_region_complete.call_args
            assert call_args[0][1] == "no_regions"


class TestSubmitRegionAsync:
    """Tests for Pipeline._submit_region_async() method."""

    @pytest.fixture
    def mock_pipeline_config(self):
        """Create a minimal PipelineConfig for testing."""
        from glmocr.config import PipelineConfig

        config = PipelineConfig()
        config.layout.use_polygon = False
        return config

    @pytest.fixture
    def mock_aggregator(self):
        """Create a mock RegionAggregator."""
        aggregator = AsyncMock()
        aggregator.on_region_complete = AsyncMock()
        return aggregator

    @pytest.mark.asyncio
    async def test_submit_region_async_skip_task_type(
        self, mock_pipeline_config, mock_aggregator
    ):
        """_submit_region_async() handles 'skip' task_type correctly."""
        from glmocr.pipeline import Pipeline

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config

            region_info = {
                "page_idx": 0,
                "page": MagicMock(),
                "region": {"task_type": "skip", "bbox_2d": [10, 10, 50, 50]},
            }

            await pipeline._submit_region_async(
                "doc123", "region_0", region_info, mock_aggregator
            )

            mock_aggregator.on_region_complete.assert_called_once()
            call_args = mock_aggregator.on_region_complete.call_args
            assert call_args[0][0] == "doc123"
            assert call_args[0][1] == "region_0"
            result = call_args[0][2]
            assert result["content"] is None

    @pytest.mark.asyncio
    async def test_submit_region_async_crop_failure(
        self, mock_pipeline_config, mock_aggregator
    ):
        """_submit_region_async() handles crop failure gracefully."""
        from glmocr.pipeline import Pipeline

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config

            region_info = {
                "page_idx": 0,
                "page": MagicMock(),
                "region": {"task_type": "text", "bbox_2d": [10, 10, 50, 50]},
            }

            with patch(
                "glmocr.pipeline.pipeline.crop_image_region",
                side_effect=RuntimeError("Crop failed"),
            ):
                await pipeline._submit_region_async(
                    "doc123", "region_0", region_info, mock_aggregator
                )

            mock_aggregator.on_region_complete.assert_called_once()
            call_args = mock_aggregator.on_region_complete.call_args
            result = call_args[0][2]
            assert result["content"] == ""

    @pytest.mark.asyncio
    async def test_submit_region_async_ocr_success(
        self, mock_pipeline_config, mock_aggregator
    ):
        """_submit_region_async() processes OCR response correctly."""
        from glmocr.pipeline import Pipeline
        from PIL import Image

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config
            pipeline.page_loader = MagicMock()
            pipeline.ocr_client = MagicMock()

            mock_page = Image.new("RGB", (100, 100))
            region_info = {
                "page_idx": 0,
                "page": mock_page,
                "region": {"task_type": "text", "bbox_2d": [10, 10, 50, 50]},
            }

            # Mock crop_image_region
            mock_cropped = Image.new("RGB", (40, 40))
            with patch(
                "glmocr.pipeline.pipeline.crop_image_region",
                return_value=mock_cropped,
            ):
                # Mock build_request_from_image
                pipeline.page_loader.build_request_from_image = MagicMock(
                    return_value={"messages": []}
                )

                # Mock OCR client to return success
                mock_response = {
                    "choices": [
                        {"message": {"content": "Recognized text content"}}
                    ]
                }
                pipeline.ocr_client.process = MagicMock(
                    return_value=(mock_response, 200)
                )

                await pipeline._submit_region_async(
                    "doc123", "region_0", region_info, mock_aggregator
                )

            mock_aggregator.on_region_complete.assert_called_once()
            call_args = mock_aggregator.on_region_complete.call_args
            result = call_args[0][2]
            assert result["content"] == "Recognized text content"

    @pytest.mark.asyncio
    async def test_submit_region_async_ocr_failure(
        self, mock_pipeline_config, mock_aggregator
    ):
        """_submit_region_async() handles OCR failure correctly."""
        from glmocr.pipeline import Pipeline
        from PIL import Image

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config
            pipeline.page_loader = MagicMock()
            pipeline.ocr_client = MagicMock()

            mock_page = Image.new("RGB", (100, 100))
            region_info = {
                "page_idx": 0,
                "page": mock_page,
                "region": {"task_type": "text", "bbox_2d": [10, 10, 50, 50]},
            }

            mock_cropped = Image.new("RGB", (40, 40))
            with patch(
                "glmocr.pipeline.pipeline.crop_image_region",
                return_value=mock_cropped,
            ):
                pipeline.page_loader.build_request_from_image = MagicMock(
                    return_value={"messages": []}
                )

                # Mock OCR client to return failure
                pipeline.ocr_client.process = MagicMock(
                    return_value=({"error": "Service unavailable"}, 500)
                )

                await pipeline._submit_region_async(
                    "doc123", "region_0", region_info, mock_aggregator
                )

            mock_aggregator.on_region_complete.assert_called_once()
            call_args = mock_aggregator.on_region_complete.call_args
            result = call_args[0][2]
            assert result["content"] is None

    @pytest.mark.asyncio
    async def test_submit_region_async_exception_marks_region_as_failed(
        self, mock_pipeline_config, mock_aggregator
    ):
        """_submit_region_async() marks region as failed on exception."""
        from glmocr.pipeline import Pipeline

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config

            region_info = {
                "page_idx": 0,
                "page": MagicMock(),
                "region": {"task_type": "text", "bbox_2d": [10, 10, 50, 50]},
            }

            # Make crop_image_region raise an unexpected exception
            with patch(
                "glmocr.pipeline.pipeline.crop_image_region",
                side_effect=Exception("Unexpected error"),
            ):
                # Even though crop failure is handled, let's test a deeper exception
                # by making the aggregator raise
                mock_aggregator.on_region_complete = AsyncMock(
                    side_effect=RuntimeError("Redis down")
                )

                # Should not raise - exception is caught and logged
                await pipeline._submit_region_async(
                    "doc123", "region_0", region_info, mock_aggregator
                )


class TestProcessAsyncBackwardCompatibility:
    """Tests to ensure process() still works (backward compatibility)."""

    @pytest.fixture
    def mock_pipeline_config(self):
        """Create a minimal PipelineConfig for testing."""
        from glmocr.config import PipelineConfig

        config = PipelineConfig()
        config.layout.use_polygon = False
        return config

    def test_process_method_exists(self, mock_pipeline_config):
        """Pipeline still has the synchronous process() method."""
        from glmocr.pipeline import Pipeline

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config

            # Verify process method exists and is callable
            assert hasattr(pipeline, "process")
            assert callable(pipeline.process)

    def test_process_is_generator(self, mock_pipeline_config):
        """process() returns a generator (not an async generator)."""
        import inspect
        from glmocr.pipeline import Pipeline

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config

            # Verify it's a regular generator function
            assert inspect.isgeneratorfunction(pipeline.process)
            # Verify process_async is an async function
            assert inspect.iscoroutinefunction(pipeline.process_async)

    def test_process_async_is_coroutine(self, mock_pipeline_config):
        """process_async() is an async coroutine function."""
        import inspect
        from glmocr.pipeline import Pipeline

        with patch.object(Pipeline, "__init__", lambda self, config, **kw: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.config = mock_pipeline_config

            assert inspect.iscoroutinefunction(pipeline.process_async)
