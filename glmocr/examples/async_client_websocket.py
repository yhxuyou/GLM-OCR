#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLM-OCR WebSocket 客户端示例

本示例展示如何通过 WebSocket 与 GLM-OCR 异步服务器交互，实现实时进度推送：
1. 提交文档进行异步处理（HTTP）
2. 通过 WebSocket 连接接收实时进度更新
3. 处理完成事件并获取最终结果
4. 错误处理与断线重连

依赖安装：
    pip install websockets requests

使用方法：
    python async_client_websocket.py <文件路径> [--server-url URL]

示例：
    python async_client_websocket.py document.pdf
    python async_client_websocket.py image.png --server-url ws://localhost:8080
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    print("[错误] 未安装 requests 库，请执行: pip install requests")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("[错误] 未安装 websockets 库，请执行: pip install websockets")
    sys.exit(1)


class GLMOCRWebSocketClient:
    """GLM-OCR WebSocket 客户端
    
    通过 WebSocket 接收实时进度更新。
    """

    def __init__(
        self,
        http_url: str = "http://localhost:5002",
        ws_url: str = "ws://localhost:5002",
        timeout: int = 30,
    ):
        """初始化客户端
        
        Args:
            http_url: HTTP API 地址（用于提交文档）
            ws_url: WebSocket 地址（用于接收进度）
            timeout: HTTP 请求超时时间（秒）
        """
        self.http_url = http_url.rstrip("/")
        self.ws_url = ws_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def submit_document(self, file_path: str) -> str:
        """提交文档进行异步处理（HTTP 请求）
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            文档 ID (doc_id)
            
        Raises:
            FileNotFoundError: 文件不存在
            requests.RequestException: HTTP 请求失败
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        url = f"{self.http_url}/parse/async"
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path_obj.name, f)}
            response = self.session.post(url, files=files, timeout=self.timeout)

        response.raise_for_status()
        result = response.json()
        
        return result["doc_id"]

    async def watch_progress(
        self,
        doc_id: str,
        on_progress: Optional[callable] = None,
        on_complete: Optional[callable] = None,
        on_error: Optional[callable] = None,
        reconnect_attempts: int = 3,
    ) -> Dict[str, Any]:
        """通过 WebSocket 监听文档处理进度
        
        Args:
            doc_id: 文档 ID
            on_progress: 进度回调，签名: callback(completed, total)
            on_complete: 完成回调，签名: callback(result)
            on_error: 错误回调，签名: callback(message)
            reconnect_attempts: 断线重连次数
            
        Returns:
            最终处理结果字典
            
        Raises:
            ValueError: 文档不存在
            ConnectionError: WebSocket 连接失败
        """
        ws_url = f"{self.ws_url}/ws/{doc_id}"
        
        for attempt in range(reconnect_attempts + 1):
            try:
                async with websockets.connect(ws_url) as websocket:
                    print(f"[WebSocket] 已连接到: {ws_url}")
                    
                    while True:
                        # 接收消息
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        msg_type = data.get("type")
                        
                        # 进度更新
                        if msg_type == "progress":
                            completed = data.get("completed", 0)
                            total = data.get("total", 0)
                            
                            if on_progress:
                                on_progress(completed, total)
                            else:
                                self._default_progress_handler(completed, total)
                        
                        # 处理完成
                        elif msg_type == "complete":
                            result = data.get("result", {})
                            
                            if on_complete:
                                on_complete(result)
                            else:
                                self._default_complete_handler(result)
                            
                            return result
                        
                        # 错误
                        elif msg_type == "error":
                            error_msg = data.get("message", "未知错误")
                            
                            if on_error:
                                on_error(error_msg)
                            else:
                                self._default_error_handler(error_msg)
                            
                            # 文档不存在等严重错误，直接抛出
                            if "not found" in error_msg.lower():
                                raise ValueError(f"文档不存在: {doc_id}")
                            
                            return {"error": error_msg}
                        
                        else:
                            print(f"[WebSocket] 未知消息类型: {msg_type}")
                            
            except websockets.exceptions.ConnectionClosed as e:
                print(f"[WebSocket] 连接关闭: {e.code} - {e.reason}")
                
                # 如果是正常关闭（完成），返回空结果
                if e.code == 1000:
                    return {}
                
                # 尝试重连
                if attempt < reconnect_attempts:
                    print(f"[WebSocket] 尝试重连 ({attempt + 1}/{reconnect_attempts})...")
                    await asyncio.sleep(2)
                else:
                    raise ConnectionError(f"WebSocket 连接失败，已重试 {reconnect_attempts} 次")
                    
            except websockets.exceptions.WebSocketException as e:
                print(f"[WebSocket] 错误: {e}")
                
                if attempt < reconnect_attempts:
                    print(f"[WebSocket] 尝试重连 ({attempt + 1}/{reconnect_attempts})...")
                    await asyncio.sleep(2)
                else:
                    raise ConnectionError(f"WebSocket 连接失败: {e}")

    def _default_progress_handler(self, completed: int, total: int):
        """默认进度处理函数"""
        if total > 0:
            percent = (completed / total) * 100
            print(f"[进度] {completed}/{total} ({percent:.1f}%)")
        else:
            print(f"[进度] {completed}/?")

    def _default_complete_handler(self, result: Dict[str, Any]):
        """默认完成处理函数"""
        print("\n[完成] 文档处理完成！")
        
        # 输出 Markdown 结果
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

    def _default_error_handler(self, message: str):
        """默认错误处理函数"""
        print(f"[错误] {message}")

    def health_check(self) -> bool:
        """健康检查"""
        try:
            url = f"{self.http_url}/health"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("status") == "ok"
        except Exception:
            return False

    def close(self):
        """关闭客户端"""
        self.session.close()


async def main_async(args):
    """异步主函数"""
    # 解析 WebSocket URL
    http_url = args.server_url
    ws_url = args.server_url
    
    # 自动转换协议
    if http_url.startswith("ws://"):
        http_url = "http://" + http_url[5:]
    elif http_url.startswith("wss://"):
        http_url = "https://" + http_url[6:]
    
    if ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[7:]
    elif ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[8:]
    
    # 创建客户端
    client = GLMOCRWebSocketClient(
        http_url=http_url,
        ws_url=ws_url,
        timeout=args.request_timeout,
    )
    
    try:
        # 1. 健康检查
        print("[信息] 检查服务器状态...")
        if not client.health_check():
            print(f"[错误] 无法连接到服务器: {http_url}")
            print("[提示] 请确保服务器已启动，可运行: ./start_async_server.sh")
            return 1
        print("[成功] 服务器连接正常")
        print()
        
        # 2. 提交文档
        print(f"[信息] 提交文档: {args.file}")
        try:
            doc_id = client.submit_document(args.file)
            print(f"[成功] 文档已提交，ID: {doc_id}")
            print()
        except FileNotFoundError as e:
            print(f"[错误] {e}")
            return 1
        except requests.RequestException as e:
            print(f"[错误] 提交文档失败: {e}")
            return 1
        
        # 3. 通过 WebSocket 监听进度
        print("[信息] 连接 WebSocket 监听进度...")
        print()
        
        try:
            result = await client.watch_progress(
                doc_id=doc_id,
                reconnect_attempts=args.reconnect_attempts,
            )
            
            # 4. 保存结果
            if result and "error" not in result:
                output_file = f"{Path(args.file).stem}_result.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\n[信息] 完整结果已保存到: {output_file}")
                return 0
            else:
                print("\n[错误] 未获取到有效结果")
                return 1
                
        except ValueError as e:
            print(f"\n[错误] {e}")
            return 1
        except ConnectionError as e:
            print(f"\n[错误] {e}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n[信息] 用户中断操作")
        return 130
    finally:
        client.close()
    
    return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GLM-OCR WebSocket 客户端示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文档
  python async_client_websocket.py document.pdf
  
  # 指定服务器地址（支持 ws:// 或 http://）
  python async_client_websocket.py image.png --server-url ws://192.168.1.100:8080
  
  # 设置重连次数
  python async_client_websocket.py document.pdf --reconnect-attempts 5
        """,
    )
    
    parser.add_argument("file", help="要处理的文档文件路径")
    parser.add_argument(
        "--server-url",
        default="http://localhost:5002",
        help="服务器地址（默认: http://localhost:5002，支持 ws:// 或 http://）",
    )
    parser.add_argument(
        "--reconnect-attempts",
        type=int,
        default=3,
        help="WebSocket 断线重连次数（默认: 3）",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=30,
        help="HTTP 请求超时秒数（默认: 30）",
    )
    
    args = parser.parse_args()
    
    # 运行异步主函数
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
