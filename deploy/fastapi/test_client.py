#!/usr/bin/env python3
"""
GLM-OCR FastAPI服务测试客户端
"""
import asyncio
import time
import argparse
from pathlib import Path
import requests


class OCRClient:
    """OCR客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> dict:
        """健康检查"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def ocr_sync(self, file_path: str, save_layout: bool = False) -> dict:
        """同步OCR"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'save_layout_visualization': str(save_layout).lower()}
            
            response = requests.post(
                f"{self.base_url}/api/v1/ocr/sync",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
    
    def ocr_async(self, file_path: str, save_layout: bool = False) -> str:
        """异步OCR，返回task_id"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'save_layout_visualization': str(save_layout).lower()}
            
            response = requests.post(
                f"{self.base_url}/api/v1/ocr/async",
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
            return result['task_id']
    
    def get_task(self, task_id: str) -> dict:
        """获取任务状态"""
        response = requests.get(f"{self.base_url}/api/v1/tasks/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_task(self, task_id: str, timeout: int = 300, poll_interval: float = 1.0) -> dict:
        """等待任务完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = self.get_task(task_id)
            
            if task['status'] in ['completed', 'failed', 'cancelled', 'timeout']:
                return task
            
            print(f"Waiting for task {task_id}... Status: {task['status']}")
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        response = requests.get(f"{self.base_url}/api/v1/stats")
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser(description='GLM-OCR API Test Client')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--mode', choices=['sync', 'async'], default='async', help='OCR mode')
    parser.add_argument('--file', required=True, help='Image/PDF file to process')
    parser.add_argument('--save-layout', action='store_true', help='Save layout visualization')
    
    args = parser.parse_args()
    
    client = OCRClient(args.url)
    
    # 健康检查
    print("Checking service health...")
    try:
        health = client.health_check()
        print(f"Service status: {health['status']}")
        print(f"Version: {health['version']}")
        if health.get('queue_stats'):
            print(f"Queue stats: {health['queue_stats']}")
    except Exception as e:
        print(f"Health check failed: {e}")
        print("Is the service running?")
        return
    
    # 检查文件
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {args.file}")
        return
    
    print(f"\nProcessing file: {file_path}")
    print(f"Mode: {args.mode}")
    print()
    
    try:
        if args.mode == 'sync':
            # 同步模式
            print("Running sync OCR...")
            result = client.ocr_sync(str(file_path), args.save_layout)
            
            if result['success']:
                print("\n✓ OCR completed successfully!")
                if result['result']:
                    markdown = result['result'].get('markdown_result', '')
                    if markdown:
                        print("\nMarkdown result:")
                        print("-" * 50)
                        print(markdown[:500])
                        if len(markdown) > 500:
                            print("... (truncated)")
                        print("-" * 50)
            else:
                print(f"\n✗ OCR failed: {result.get('error')}")
        
        else:
            # 异步模式
            print("Running async OCR...")
            task_id = client.ocr_async(str(file_path), args.save_layout)
            print(f"Task created: {task_id}")
            
            print("\nWaiting for completion...")
            task = client.wait_for_task(task_id)
            
            if task['status'] == 'completed':
                print("\n✓ OCR completed successfully!")
                if task['result']:
                    markdown = task['result'].get('markdown_result', '')
                    if markdown:
                        print("\nMarkdown result:")
                        print("-" * 50)
                        print(markdown[:500])
                        if len(markdown) > 500:
                            print("... (truncated)")
                        print("-" * 50)
            else:
                print(f"\n✗ Task failed with status: {task['status']}")
                if task.get('error'):
                    print(f"Error: {task['error']}")
        
        # 获取统计信息
        print("\nService stats:")
        stats = client.get_stats()
        print(f"Queue: {stats.get('queue', {})}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
