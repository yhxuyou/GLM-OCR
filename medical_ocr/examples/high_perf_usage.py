"""
Medical OCR - High Performance Server Example.

Demonstrates:
- Using PipelinePool for high concurrency
- Single GPU vs Multi-GPU configurations
- Performance monitoring
"""

import os
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, '/workspace/medical_ocr')

from medical_ocr import (
    MedicalOcrPipeline,
    create_pipeline_pool,
)
from medical_ocr.utils import setup_logging
from glmocr.config import load_config


def demo_single_gpu_pool():
    """Demo: Single GPU with multiple pipelines."""
    
    print("=" * 70)
    print("High Performance - Single GPU Pool Demo")
    print("=" * 70)
    
    config = load_config()
    
    print("\n1. Creating Single-GPU Pipeline Pool...")
    print("   - 4 Pipeline instances on GPU 0")
    print("   - Total GPU memory shared")
    
    pool = create_pipeline_pool(
        pipeline_config=config.pipeline,
        pool_size=4,
        mode="single_gpu",
        gpu_device_id=0,
        gpu_memory_fraction=0.9,
    )
    
    print("   ✅ Pool created")
    
    print("\n2. Initializing pool...")
    pool.initialize()
    print("   ✅ Pool initialized")
    
    # Benchmark
    print("\n3. Benchmark:")
    print("-" * 70)
    
    num_requests = 8
    print(f"   Processing {num_requests} requests...")
    
    requests = [
        {"images": [f"/tmp/doc_{i}.jpg"]}
        for i in range(num_requests)
    ]
    
    start_time = time.time()
    
    try:
        results = pool.process_batch(requests)
        elapsed = time.time() - start_time
        
        print(f"   ✅ Completed in {elapsed:.2f}s")
        print(f"   ✅ Throughput: {num_requests / elapsed:.2f} req/s")
    except Exception as e:
        print(f"   ⚠️ Benchmark error: {e}")
        print("   (This is expected if glmocr is not fully installed)")
    
    print("\n4. Shutting down...")
    pool.shutdown()
    print("   ✅ Pool shut down")
    
    print("\n" + "=" * 70)


def demo_multi_gpu_pool():
    """Demo: Multiple GPUs with dedicated pipelines."""
    
    print("\n" + "=" * 70)
    print("High Performance - Multi-GPU Pool Demo")
    print("=" * 70)
    
    config = load_config()
    
    print("\n1. Creating Multi-GPU Pipeline Pool...")
    print("   - 4 workers across GPUs 0 and 1")
    print("   - Each GPU gets 2 dedicated pipelines")
    
    pool = create_pipeline_pool(
        pipeline_config=config.pipeline,
        pool_size=4,
        mode="multi_gpu",
        gpu_device_ids=[0, 1],
    )
    
    print("   ✅ Pool created")
    print("   Note: This would require multiple GPUs to actually initialize")
    
    pool.shutdown()
    
    print("\n" + "=" * 70)


def demo_pool_config():
    """Demo: Pool configuration options."""
    
    print("\n" + "=" * 70)
    print("Pipeline Pool - Configuration Guide")
    print("=" * 70)
    
    print("\n1. Single GPU Mode (Recommended for most cases):")
    print("-" * 70)
    print("""
    pool = create_pipeline_pool(
        pipeline_config=config.pipeline,
        pool_size=4,              # Number of pipelines
        mode="single_gpu",         # Use single GPU mode
        gpu_device_id=0,          # Which GPU to use
        gpu_memory_fraction=0.9,  # Max GPU memory (optional)
    )
    """)
    
    print("\n2. Multi-GPU Mode (For heavy workloads):")
    print("-" * 70)
    print("""
    pool = create_pipeline_pool(
        pipeline_config=config.pipeline,
        pool_size=8,                      # Total workers
        mode="multi_gpu",                  # Use multi GPU mode
        gpu_device_ids=[0, 1, 2, 3],       # Available GPUs
    )
    """)
    
    print("\n3. Pool Size Guidelines:")
    print("-" * 70)
    print("""
    | GPU Memory | Recommended Pool Size |
    |------------|------------------------|
    | 8GB        | 2-4                     |
    | 12GB       | 4-6                     |
    | 24GB       | 6-8                     |
    | 48GB+      | 8-16                    |
    """)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    setup_logging("INFO")
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "config"],
        default="single",
        help="Demo mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        demo_single_gpu_pool()
    elif args.mode == "multi":
        demo_multi_gpu_pool()
    elif args.mode == "config":
        demo_pool_config()
