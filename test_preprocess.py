"""Test script for the image preprocessing service."""

import os
import sys
import time
import base64
import io
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from preprocess_client import PreprocessClient


def create_test_image(rotate_degrees: int = 0):
    """Create a test image with text for orientation testing."""
    img = Image.new('RGB', (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    text = "GLM-OCR PREPROCESSING TEST\n\nDocument detection and orientation test.\n\nABCDEFGHIJKLMNOP\n0123456789\n\nThis is a test document for preprocessing."
    
    y = 50
    for line in text.split('\n'):
        draw.text((50, y), line, fill=(50, 50, 50))
        y += 30
    
    if rotate_degrees != 0:
        img = img.rotate(rotate_degrees, expand=True)
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def test_sync_preprocessing():
    """Test synchronous preprocessing endpoint."""
    print("=" * 60)
    print("TEST: Synchronous Preprocessing")
    print("=" * 60)
    
    client = PreprocessClient("http://localhost:7001")
    
    test_images = [
        ("Original (0°)", 0),
        ("Rotated 90°", 90),
        ("Rotated 180°", 180),
        ("Rotated 270°", 270),
    ]
    
    for name, rotation in test_images:
        print(f"\nProcessing: {name}")
        b64_img = create_test_image(rotation)
        
        try:
            result = client.preprocess_sync(f"data:image/jpeg;base64,{b64_img}")
            
            doc_detected = result.get("document_detected", False)
            orientation = result.get("orientation_angle", 0)
            deskewed = result.get("deskewed_angle", 0.0)
            confidence = result.get("confidence", 0.0)
            has_corrected = "corrected_image" in result
            
            print(f"  Document detected: {doc_detected}")
            print(f"  Detected orientation: {orientation}°")
            print(f"  Deskew angle: {deskewed:.2f}°")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Corrected image: {'Yes' if has_corrected else 'No'}")
            
            assert doc_detected, "Document should be detected"
            assert abs(orientation - rotation) % 360 == 0, f"Orientation mismatch: expected {rotation}, got {orientation}"
            print(f"  [PASS]")
            
        except Exception as e:
            print(f"  [FAIL] {e}")


def test_async_preprocessing():
    """Test asynchronous preprocessing endpoint."""
    print("\n" + "=" * 60)
    print("TEST: Asynchronous Preprocessing")
    print("=" * 60)
    
    client = PreprocessClient("http://localhost:7001")
    
    b64_img = create_test_image(180)
    job_id = client.preprocess_async(f"data:image/jpeg;base64,{b64_img}")
    print(f"Submitted job: {job_id[:12]}...")
    
    status = client.get_status(job_id)
    print(f"Initial status: {status['status']}")
    
    result = client.wait_for_result(job_id, timeout=30)
    print(f"Final status: {result['status']}")
    
    if result["status"] == "completed":
        print(f"  Orientation detected: {result['result']['orientation_angle']}°")
        print(f"  Document detected: {result['result']['document_detected']}")
        print("  [PASS]")
    else:
        print(f"  [FAIL] {result.get('error', 'Unknown error')}")


def test_batch_preprocessing():
    """Test batch preprocessing endpoint."""
    print("\n" + "=" * 60)
    print("TEST: Batch Preprocessing")
    print("=" * 60)
    
    client = PreprocessClient("http://localhost:7001")
    
    images = [create_test_image(i * 90) for i in range(4)]
    images_b64 = [f"data:image/jpeg;base64,{img}" for img in images]
    
    result = client.batch_preprocess(images_b64)
    print(f"Accepted: {result['accepted']}/{len(images)}")
    
    for res in result["results"]:
        print(f"  Job {res['job_id'][:12]}...: {'accepted' if res['accepted'] else 'rejected'}")
    
    print("  [PASS]")


def test_health_and_stats():
    """Test health and stats endpoints."""
    print("\n" + "=" * 60)
    print("TEST: Health and Stats")
    print("=" * 60)
    
    client = PreprocessClient("http://localhost:7001")
    
    health = client.health()
    print(f"Health status: {health['status']}")
    print(f"Active workers: {health['stats']['active_workers']}")
    print(f"Queue size: {health['stats']['queue_size']}")
    
    stats = client.stats()
    print(f"Total submitted: {stats['pool']['submitted']}")
    print(f"Total completed: {stats['pool']['completed']}")
    
    print("  [PASS]")


def main():
    print("\n" + "#" * 60)
    print("# GLM-OCR Preprocessing Service Test Suite")
    print("#" * 60)
    print()
    
    import subprocess
    import requests
    
    try:
        requests.get("http://localhost:7001/health", timeout=5)
    except Exception:
        print("Starting preprocessing server...")
        subprocess.Popen([sys.executable, "preprocess_server.py", "--port", "7001"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)
    
    try:
        test_sync_preprocessing()
        test_async_preprocessing()
        test_batch_preprocessing()
        test_health_and_stats()
        
        print("\n" + "#" * 60)
        print("# ALL TESTS COMPLETED")
        print("#" * 60)
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    from PIL import ImageDraw
    main()