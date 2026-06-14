#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLM-OCR 异步客户端示例（HTTP 轮询方式）

本示例展示如何通过 HTTP API 与 GLM-OCR 异步服务器交互：
1. 提交文档进行异步处理
2. 轮询处理状态
3. 获取最终结果
4. 错误处理

依赖安装：
    pip install requests

使用方法：
    python async_client_example.py <文件路径> [--server-url URL]

示例：
    python async_client_example.py document.pdf
    python async_client_example.py image.png --server-url http://localhost:8080
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    print("[错误] 未安装 requests 库，请执行: pip install requests")
    sys.exit(1)


class GLMOCRAsyncClient:
    """GLM-OCR 异步客户端
    
    封装了与异步服务器交互的所有 HTTP 请求。
    """

    def __init__(self, server_url: str = "http://localhost:5002", timeout: int = 30):
        """初始化客户端
        
        Args:
            server_url: 服务器地址（默认: http://localhost:5002）
            timeout: 请求超时时间（秒）
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def submit_document(self, file_path: str) -> Dict[str, Any]:
        """提交文档进行异步处理
        
        Args:
            file_path: 文档文件路径（支持图片、PDF 等）
            
        Returns:
            包含 doc_id 和 status 的字典
            示例: {"doc_id": "uuid-string", "status": "processing"}
            
        Raises:
            FileNotFoundError: 文件不存在
            requests.RequestException: HTTP 请求失败
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        url = f"{self.server_url}/parse/async"
        
        # 读取文件并以 multipart/form-data 格式上传
        with open(file_path, "rb") as f:
            files = {"file": (file_path_obj.name, f)}
            response = self.session.post(url, files=files, timeout=self.timeout)

        # 检查响应状态
        response.raise_for_status()
        
        result = response.json()
        return result

    def get_status(self, doc_id: str) -> Dict[str, Any]:
        """查询文档处理状态
        
        Args:
            doc_id: 文档 ID（由 submit_document 返回）
            
        Returns:
            包含处理进度的字典
            示例: {
                "completed": 5,      # 已完成的区域数
                "total": 10,         # 总区域数
                "status": "processing"  # processing / completed / not_found
            }
            
        Raises:
            requests.RequestException: HTTP 请求失败
        """
        url = f"{self.server_url}/parse/status/{doc_id}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        return result

    def get_result(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档处理的最终结果
        
        Args:
            doc_id: 文档 ID
            
        Returns:
            如果处理完成，返回包含 OCR 结果的字典；
            如果仍在处理，返回 None
            
        Raises:
            requests.RequestException: HTTP 请求失败（404 表示文档不存在）
        """
        url = f"{self.server_url}/parse/result/{doc_id}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            
            # 202 表示仍在处理中
            if response.status_code == 202:
                return None
            
            # 其他错误状态码
            response.raise_for_status()
            
            result = response.json()
            return result
            
        except requests.exceptions.HTTPError as e:
            # 404 表示文档不存在
            if e.response.status_code == 404:
                raise ValueError(f"文档不存在: {doc_id}")
            raise

    def wait_for_completion(
        self,
        doc_id: str,
        poll_interval: float = 2.0,
        max_wait_time: Optional[float] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """等待文档处理完成
        
        Args:
            doc_id: 文档 ID
            poll_interval: 轮询间隔（秒，默认 2 秒）
            max_wait_time: 最大等待时间（秒，None 表示无限等待）
            progress_callback: 进度回调函数，签名: callback(completed, total)
            
        Returns:
            最终处理结果字典
            
        Raises:
            TimeoutError: 超过最大等待时间
            ValueError: 文档不存在
        """
        start_time = time.time()
        last_progress = -1
        
        while True:
            # 检查超时
            if max_wait_time is not None:
                elapsed = time.time() - start_time
                if elapsed > max_wait_time:
                    raise TimeoutError(f"等待超时（已等待 {elapsed:.1f} 秒）")
            
            # 查询状态
            status = self.get_status(doc_id)
            
            # 文档不存在
            if status["status"] == "not_found":
                raise ValueError(f"文档不存在: {doc_id}")
            
            # 处理完成
            if status["status"] == "completed":
                # 获取最终结果
                result = self.get_result(doc_id)
                if result is not None:
                    return result
                # 如果结果为 None，继续轮询（可能是状态更新延迟）
                time.sleep(poll_interval)
                continue
            
            # 仍在处理中，显示进度
            completed = status.get("completed", 0)
            total = status.get("total", 0)
            
            if completed != last_progress:
                if progress_callback:
                    progress_callback(completed, total)
                else:
                    print(f"[进度] {completed}/{total} ({completed/total*100:.1f}%)" if total > 0 else f"[进度] {completed}/?")
                last_progress = completed
            
            # 等待后再次轮询
            time.sleep(poll_interval)

    def health_check(self) -> bool:
        """健康检查
        
        Returns:
            True 表示服务器正常，False 表示异常
        """
        try:
            url = f"{self.server_url}/health"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("status") == "ok"
        except Exception:
            return False

    def close(self):
        """关闭客户端连接"""
        self.session.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GLM-OCR 异步客户端示例（HTTP 轮询方式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文档
  python async_client_example.py document.pdf
  
  # 指定服务器地址
  python async_client_example.py image.png --server-url http://192.168.1.100:8080
  
  # 设置轮询间隔和超时
  python async_client_example.py document.pdf --poll-interval 3 --timeout 300
        """,
    )
    
    parser.add_argument("file", help="要处理的文档文件路径")
    parser.add_argument(
        "--server-url",
        default="http://localhost:5002",
        help="服务器地址（默认: http://localhost:5002）",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="轮询间隔秒数（默认: 2.0）",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=None,
        help="最大等待时间秒数（默认: 无限等待）",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=30,
        help="HTTP 请求超时秒数（默认: 30）",
    )
    
    args = parser.parse_args()
    
    # 创建客户端
    client = GLMOCRAsyncClient(
        server_url=args.server_url,
        timeout=args.request_timeout,
    )
    
    try:
        # 1. 健康检查
        print("[信息] 检查服务器状态...")
        if not client.health_check():
            print(f"[错误] 无法连接到服务器: {args.server_url}")
            print("[提示] 请确保服务器已启动，可运行: ./start_async_server.sh")
            sys.exit(1)
        print("[成功] 服务器连接正常")
        print()
        
        # 2. 提交文档
        print(f"[信息] 提交文档: {args.file}")
        try:
            submit_result = client.submit_document(args.file)
            doc_id = submit_result["doc_id"]
            print(f"[成功] 文档已提交，ID: {doc_id}")
            print()
        except FileNotFoundError as e:
            print(f"[错误] {e}")
            sys.exit(1)
        except requests.RequestException as e:
            print(f"[错误] 提交文档失败: {e}")
            sys.exit(1)
        
        # 3. 等待处理完成
        print("[信息] 等待处理完成...")
        try:
            result = client.wait_for_completion(
                doc_id=doc_id,
                poll_interval=args.poll_interval,
                max_wait_time=args.max_wait,
            )
            print()
            print("[成功] 文档处理完成！")
            print()
            
            # 4. 输出结果
            print("=" * 60)
            print("处理结果:")
            print("=" * 60)
            
            # 输出 Markdown 结果（如果存在）
            if "markdown_result" in result and result["markdown_result"]:
                print("\n[Markdown 输出]:")
                print("-" * 60)
                print(result["markdown_result"])
                print("-" * 60)
            
            # 输出 JSON 结果摘要
            if "json_result" in result and result["json_result"]:
                print(f"\n[JSON 结果]: 共 {len(result['json_result'])} 页")
                for page_idx, page_data in enumerate(result["json_result"]):
                    if isinstance(page_data, list):
                        print(f"  第 {page_idx + 1} 页: {len(page_data)} 个区域")
            
            # 保存完整结果到文件
            output_file = f"{Path(args.file).stem}_result.json"
            import json
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n[信息] 完整结果已保存到: {output_file}")
            
        except TimeoutError as e:
            print(f"\n[错误] {e}")
            print(f"[提示] 可以使用 --max-wait 参数延长等待时间")
            sys.exit(1)
        except ValueError as e:
            print(f"\n[错误] {e}")
            sys.exit(1)
        except requests.RequestException as e:
            print(f"\n[错误] 查询状态失败: {e}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n[信息] 用户中断操作")
        sys.exit(130)
    finally:
        client.close()


if __name__ == "__main__":
    main()
