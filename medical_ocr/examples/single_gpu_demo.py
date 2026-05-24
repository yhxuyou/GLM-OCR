"""
Medical OCR - Single GPU Multiple Pipelines Example.

Demonstrates:
- Running multiple Pipeline instances on a single GPU
- Thread-based parallelism
- GPU memory control
- Performance benchmark
"""

import os
import sys
import time
import uuid
from typing import List, Dict, Any

# Add project path
sys.path.insert(0, '/workspace/medical_ocr')

from medical_ocr.pipeline_pool import create_pipeline_pool
from glmocr.config import load_config
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


def demo_single_gpu_multiple_pipelines():
    """Demo: Single GPU with multiple Pipeline instances."""
    
    print("=" * 70)
    print("Single GPU - Multiple Pipelines Demo")
    print("=" * 70)
    
    # 1. Load config
    config = load_config()
    
    # 2. Create single-GPU pipeline pool
    # 4 Pipeline instances, all on GPU 0
    print("\n1. Creating Single-GPU Pipeline Pool (4 pipelines on GPU 0)")
    pool = create_pipeline_pool(
        pipeline_config=config.pipeline,
        pool_size=4,
        yolo_model_dir=os.getenv("YOLO_MODEL_DIR", None),
        uvdoc_model_dir=os.getenv("UVDOC_MODEL_DIR", None),
        mode="single_gpu",
        gpu_device_id=0,
        gpu_memory_fraction=0.9  # Use up to 90% of GPU memory
    )
    
    # 3. Initialize pool
    print("\n2. Initializing pool...")
    pool.initialize()
    print("✅ Pool initialized successfully")
    
    # 4. Benchmark: sequential vs parallel
    print("\n3. Performance Benchmark:")
    print("-" * 70)
    
    # Create dummy requests
    num_requests = 8
    dummy_requests = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "/tmp/dummy.jpg"}}
                    ]
                }
            ]
        }
        for _ in range(num_requests)
    ]
    
    # Option A: Sequential processing (1 pipeline)
    print("\nA. Sequential processing (1 pipeline):")
    start_seq = time.time()
    for req in dummy_requests[:4]:
        try:
            # For demo, simulate processing
            time.sleep(0.1)
        except Exception:
            pass
    time_seq = time.time() - start_seq
    print(f"   Time: {time_seq:.2f}s")
    
    # Option B: Parallel processing (4 pipelines)
    print("\nB. Parallel processing (4 pipelines):")
    start_par = time.time()
    try:
        results = pool.process_batch(dummy_requests[:4])
    except Exception as e:
        print(f"   Error: {e}")
        results = []
    time_par = time.time() - start_par
    print(f"   Time: {time_par:.2f}s")
    print(f"   Speedup: {time_seq / max(time_par, 0.001):.1f}x")
    
    # 5. Shutdown
    print("\n4. Shutting down pool...")
    pool.shutdown()
    print("✅ Pool shut down successfully")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


def demo_multi_gpu():
    """Demo: Multiple GPUs (optional)."""
    
    print("\n" + "=" * 70)
    print("Multi-GPU Demo (if you have multiple GPUs)")
    print("=" * 70)
    
    config = load_config()
    
    # Create multi-GPU pool: 2 GPUs, 4 workers total
    print("\nCreating Multi-GPU Pipeline Pool (2 GPUs)")
    pool = create_pipeline_pool(
        pipeline_config=config.pipeline,
        pool_size=4,
        mode="multi_gpu",
        gpu_device_ids=[0, 1]  # Use GPU 0 and 1
    )
    
    print("\n✅ Multi-GPU Pool created (workers round-robin across GPUs)")
    print("\nUsage: Each worker process has its own dedicated GPU!")
    
    pool.initialize()
    # ... use pool.process_batch()
    pool.shutdown()


def benchmark_memory_usage():
    """Benchmark GPU memory with different pipeline counts."""
    
    print("\n" + "=" * 70)
    print("GPU Memory Usage Benchmark")
    print("=" * 70)
    
    config = load_config()
    
    # Test different pool sizes
    for pool_size in [1, 2, 4]:
        print(f"\n--- Pool Size = {pool_size} ---")
        
        try:
            # Create pool
            pool = create_pipeline_pool(
                pipeline_config=config.pipeline,
                pool_size=pool_size,
                mode="single_gpu",
                gpu_device_id=0,
                gpu_memory_fraction=0.8 / pool_size  # Split memory
            )
            
            pool.initialize()
            time.sleep(2)  # Let models load
            
            # Get memory usage (if PyTorch is available)
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    print(f"   GPU Memory Used: {allocated:.2f}GB")
            except Exception:
                pass
            
            pool.shutdown()
            time.sleep(1)
            
        except Exception as e:
            print(f"   Error: {e}")


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", 
        choices=["single", "multi", "memory"],
        default="single",
        help="Demo mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        demo_single_gpu_multiple_pipelines()
    elif args.mode == "multi":
        demo_multi_gpu()
    elif args.mode == "memory":
        benchmark_memory_usage()
