"""
Medical OCR - Quick Start Example.

This example demonstrates the basic usage of Medical OCR.
"""

import os
import sys

sys.path.insert(0, '/workspace/medical_ocr')

from medical_ocr import (
    MedicalPageLoader,
    MedicalLayoutDetector,
    MedicalResultFormatter,
    MedicalOcrPipeline,
)
from medical_ocr.utils import setup_logging, SimpleCache

from glmocr.config import load_config
from glmocr.dataloader import PageLoaderConfig


def basic_usage():
    """Basic usage example."""
    
    setup_logging("INFO")
    
    print("=" * 70)
    print("Medical OCR - Basic Usage Example")
    print("=" * 70)
    
    # 1. Load configuration
    config = load_config()
    print("\n1. Configuration loaded")
    
    # 2. Create components
    print("\n2. Creating Medical OCR components...")
    
    # MedicalPageLoader with preprocessing
    page_loader = MedicalPageLoader(
        config=PageLoaderConfig(),
        yolo_model_dir=None,  # Set path if available
        uvdoc_model_dir=None,  # Set path if available
        enable_preprocessing=True,
    )
    
    # MedicalLayoutDetector
    layout_detector = MedicalLayoutDetector(
        config=config.pipeline.layout,
    )
    
    # MedicalResultFormatter with RapidOCR coordinate extraction
    result_formatter = MedicalResultFormatter(
        config=config.pipeline.result_formatter,
    )
    
    print("   ✅ Components created")
    
    # 3. Create pipeline
    print("\n3. Creating MedicalOcrPipeline...")
    
    pipeline = MedicalOcrPipeline(
        config=config.pipeline,
        page_loader=page_loader,
        layout_detector=layout_detector,
        result_formatter=result_formatter,
    )
    
    print("   ✅ Pipeline created")
    
    # 4. Start pipeline
    print("\n4. Starting pipeline...")
    pipeline.start()
    print("   ✅ Pipeline started")
    
    # 5. Process example
    print("\n5. Processing example...")
    
    example_images = [
        "/path/to/medical_document_1.jpg",
        "/path/to/medical_document_2.jpg",
    ]
    
    try:
        # For demo, we just show the structure
        print(f"\n   Would process {len(example_images)} images")
        print("   (Set real image paths to process)")
    except Exception as e:
        print(f"   Processing error: {e}")
    
    # 6. Stop pipeline
    print("\n6. Stopping pipeline...")
    pipeline.stop()
    print("   ✅ Pipeline stopped")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


def with_cache():
    """Example with caching."""
    
    print("\n" + "=" * 70)
    print("Medical OCR - Caching Example")
    print("=" * 70)
    
    # Create cache
    cache = SimpleCache(max_size=1000, ttl=3600)
    
    # Simulate request
    request_data = {"images": ["test.jpg"]}
    
    # First request (cache miss)
    print("\n1. First request (cache miss)...")
    result = cache.get(request_data)
    if result is None:
        print("   Cache miss - processing...")
        result = {"text": "Medical record content"}
        cache.set(request_data, result)
        print("   Result cached")
    
    # Second request (cache hit)
    print("\n2. Second request (cache hit)...")
    result = cache.get(request_data)
    if result is not None:
        print("   Cache hit - using cached result!")
    
    # Print cache stats
    print("\n3. Cache statistics:")
    stats = cache.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    basic_usage()
    with_cache()
