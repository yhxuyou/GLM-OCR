"""
Medical OCR - Configuration Usage Example.

Demonstrates:
- Loading the extended config.yaml
- Using the medical-specific settings
- Initializing the pipeline with the config
"""

import os
import sys
from typing import Dict, Any

sys.path.insert(0, '/workspace/medical_ocr')

from glmocr.config import load_config
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


def load_medical_config(config_path: str = None):
    """
    Load Medical OCR configuration.
    
    Looks for config in this order:
    1. Provided path
    2. ./config.yaml
    3. ./config_simple.yaml
    4. ../config.yaml
    """
    
    # Try default locations
    default_paths = [
        config_path,
        "./config.yaml",
        "./config_simple.yaml",
        "../config.yaml",
        "/workspace/medical_ocr/config.yaml"
    ]
    
    config = None
    for path in default_paths:
        if path and os.path.exists(path):
            logger.info(f"Loading config from: {path}")
            config = load_config(path)
            break
    
    if not config:
        logger.warning("No config found, using defaults")
        config = load_config()
    
    return config


def demo_config_usage():
    """Demo: How to use the medical config."""
    
    print("=" * 70)
    print("Medical OCR Configuration Demo")
    print("=" * 70)
    
    # 1. Load config
    print("\n1. Loading configuration...")
    config = load_medical_config()
    
    # 2. Print key settings
    print("\n2. Key configuration settings:")
    print("-" * 70)
    
    # Original server settings
    print(f"\nServer:")
    print(f"  Host: {config.server.host}")
    print(f"  Port: {config.server.port}")
    
    # MaaS mode
    print(f"\nMaaS Mode:")
    print(f"  Enabled: {config.pipeline.maas.enabled}")
    if config.pipeline.maas.api_key:
        print(f"  API Key: {'*' * 16}")
    
    # Medical page loader settings
    print(f"\nMedical Page Loader:")
    print(f"  Enable Preprocessing: {config.pipeline.medical_page_loader.enable_preprocessing}")
    print(f"  YOLO Model Dir: {config.pipeline.medical_page_loader.yolo_model_dir}")
    print(f"  YOLO Threshold: {config.pipeline.medical_page_loader.yolo_confidence_threshold}")
    print(f"  RapidOCR Enabled: {config.pipeline.medical_page_loader.rapidocr_enabled}")
    print(f"  UVDoc Model Dir: {config.pipeline.medical_page_loader.uvdoc_model_dir}")
    
    # Pipeline pool settings
    print(f"\nPipeline Pool:")
    print(f"  Mode: {config.pipeline_pool.mode}")
    print(f"  Pool Size: {config.pipeline_pool.pool_size}")
    
    if config.pipeline_pool.mode == "single_gpu":
        print(f"  GPU Device ID: {config.pipeline_pool.single_gpu.gpu_device_id}")
    elif config.pipeline_pool.mode == "multi_gpu":
        print(f"  GPU Device IDs: {config.pipeline_pool.multi_gpu.gpu_device_ids}")
    
    # Cache settings
    print(f"\nCache:")
    print(f"  In-Memory: {config.cache.in_memory.enabled}")
    print(f"  Redis: {config.cache.redis.enabled}")
    
    print("\n" + "=" * 70)


def demo_create_pipeline_pool():
    """Demo: Create pipeline pool from config."""
    
    print("\n" + "=" * 70)
    print("Creating Pipeline Pool from Config")
    print("=" * 70)
    
    config = load_medical_config()
    
    # Import our pool
    from medical_ocr.pipeline_pool import create_pipeline_pool
    
    # Extract settings from config
    mode = config.pipeline_pool.mode
    pool_size = config.pipeline_pool.pool_size
    
    if mode == "single_gpu":
        gpu_device_id = config.pipeline_pool.single_gpu.gpu_device_id
        gpu_memory_fraction = config.pipeline_pool.single_gpu.gpu_memory_fraction
        
        print(f"\nCreating Single-GPU Pool (GPU {gpu_device_id}, size {pool_size})")
        
        pool = create_pipeline_pool(
            pipeline_config=config.pipeline,
            pool_size=pool_size,
            mode="single_gpu",
            gpu_device_id=gpu_device_id,
            gpu_memory_fraction=gpu_memory_fraction,
            yolo_model_dir=config.pipeline.medical_page_loader.yolo_model_dir,
            uvdoc_model_dir=config.pipeline.medical_page_loader.uvdoc_model_dir
        )
        
    elif mode == "multi_gpu":
        gpu_device_ids = config.pipeline_pool.multi_gpu.gpu_device_ids
        
        print(f"\nCreating Multi-GPU Pool (GPUs {gpu_device_ids}, size {pool_size})")
        
        pool = create_pipeline_pool(
            pipeline_config=config.pipeline,
            pool_size=pool_size,
            mode="multi_gpu",
            gpu_device_ids=gpu_device_ids,
            yolo_model_dir=config.pipeline.medical_page_loader.yolo_model_dir,
            uvdoc_model_dir=config.pipeline.medical_page_loader.uvdoc_model_dir
        )
    
    print("\n✅ Pipeline pool created from config!")
    print("\n(Note: This is a demo, pool not actually initialized in this script)")


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--demo",
        choices=["config", "pool"],
        default="config",
        help="Demo mode"
    )
    
    args = parser.parse_args()
    
    if args.demo == "config":
        demo_config_usage()
    elif args.demo == "pool":
        demo_create_pipeline_pool()
