# GLM-OCR 异步示例

本目录包含 GLM-OCR 异步服务器的使用示例，展示如何启动服务器以及通过 HTTP 和 WebSocket 与服务器交互。

## 文件说明

| 文件 | 说明 |
|------|------|
| `start_async_server.sh` | Bash 启动脚本，用于启动 GLM-OCR 异步 FastAPI 服务器 |
| `async_client_example.py` | HTTP 客户端示例，展示如何提交文档、轮询状态、获取结果 |
| `async_client_websocket.py` | WebSocket 客户端示例，展示如何接收实时进度更新 |

## API 端点

服务器启动后提供以下 API 端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/parse/async` | 提交文档进行异步处理（multipart/form-data） |
| `GET` | `/parse/status/{doc_id}` | 查询文档处理进度 |
| `GET` | `/parse/result/{doc_id}` | 获取最终处理结果 |
| `WS` | `/ws/{doc_id}` | WebSocket 实时进度推送 |
| `GET` | `/health` | 健康检查 |

## 快速开始

### 1. 安装依赖

```bash
# 安装 GLM-OCR 服务器依赖
pip install 'glmocr[server]'

# 安装客户端示例依赖
pip install requests websockets
```

### 2. 启动服务器

```bash
# 使用默认配置（0.0.0.0:5002）
./start_async_server.sh

# 自定义端口和日志级别
./start_async_server.sh --port 8080 --log-level DEBUG

# 使用自定义配置文件
./start_async_server.sh --config /path/to/config.yaml
```

**环境变量配置示例**（可在 `.env` 文件中设置）：

```bash
# 服务器配置
GLMOCR_SERVER_HOST=0.0.0.0
GLMOCR_SERVER_PORT=5002

# 智谱 API Key（MaaS 模式必需）
ZHIPU_API_KEY=your-api-key-here

# 运行模式：maas（云端）或 selfhosted（本地 GPU）
GLMOCR_MODE=maas

# Redis 连接地址
GLMOCR_REDIS_URL=redis://localhost:6379/0
```

### 3. 使用 HTTP 客户端

```bash
# 处理单个文档
python async_client_example.py document.pdf

# 指定服务器地址
python async_client_example.py image.png --server-url http://localhost:8080

# 设置轮询间隔和超时
python async_client_example.py document.pdf --poll-interval 3 --max-wait 300
```

### 4. 使用 WebSocket 客户端

```bash
# 处理文档并接收实时进度
python async_client_websocket.py document.pdf

# 指定服务器地址
python async_client_websocket.py image.png --server-url ws://localhost:8080

# 设置重连次数
python async_client_websocket.py document.pdf --reconnect-attempts 5
```

## HTTP 客户端示例详解

### 基本用法

```python
from async_client_example import GLMOCRAsyncClient

# 创建客户端
client = GLMOCRAsyncClient(server_url="http://localhost:5002")

# 1. 提交文档
result = client.submit_document("document.pdf")
doc_id = result["doc_id"]  # 如: "550e8400-e29b-41d4-a716-446655440000"

# 2. 查询状态
status = client.get_status(doc_id)
print(f"进度: {status['completed']}/{status['total']}")

# 3. 等待完成并获取结果
final_result = client.wait_for_completion(doc_id)

# 4. 关闭客户端
client.close()
```

### 响应格式

**提交响应**：
```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing"
}
```

**状态查询响应**：
```json
{
  "completed": 5,
  "total": 10,
  "status": "processing"
}
```

**最终结果响应**：
```json
{
  "json_result": [[{"index": 0, "label": "text", "content": "...", "bbox_2d": [...]}]],
  "markdown_result": "# 文档标题\n\n正文内容...",
  "original_images": ["document.pdf"]
}
```

## WebSocket 客户端示例详解

### 基本用法

```python
import asyncio
from async_client_websocket import GLMOCRWebSocketClient

async def main():
    client = GLMOCRWebSocketClient(
        http_url="http://localhost:5002",
        ws_url="ws://localhost:5002",
    )
    
    # 1. 提交文档（HTTP）
    doc_id = client.submit_document("document.pdf")
    print(f"文档 ID: {doc_id}")
    
    # 2. 监听进度（WebSocket）
    result = await client.watch_progress(doc_id)
    print("处理完成！")
    
    client.close()

asyncio.run(main())
```

### WebSocket 消息格式

**进度消息**：
```json
{
  "type": "progress",
  "completed": 5,
  "total": 10
}
```

**完成消息**：
```json
{
  "type": "complete",
  "result": {
    "json_result": [...],
    "markdown_result": "..."
  }
}
```

**错误消息**：
```json
{
  "type": "error",
  "message": "Document not found"
}
```

## 自定义回调函数

WebSocket 客户端支持自定义回调函数：

```python
async def my_progress(completed, total):
    print(f"自定义进度: {completed}/{total}")

async def my_complete(result):
    print("自定义完成处理")
    # 保存到数据库或其他存储

result = await client.watch_progress(
    doc_id=doc_id,
    on_progress=my_progress,
    on_complete=my_complete,
    on_error=lambda msg: print(f"自定义错误: {msg}"),
)
```

## 注意事项

1. **Redis 依赖**：异步服务器需要 Redis 作为后端存储，请确保 Redis 服务已启动。
2. **大文件处理**：对于大型 PDF 文件，处理时间可能较长，建议设置合理的 `max_wait` 参数。
3. **并发请求**：服务器支持并发处理多个文档，每个文档有独立的 `doc_id`。
4. **WebSocket 断线**：客户端支持自动重连，可通过 `reconnect_attempts` 参数控制重试次数。
5. **结果保存**：两个客户端示例都会将完整结果保存为 `{文件名}_result.json`。

## 常见问题

**Q: 启动服务器时提示缺少依赖？**
A: 请确保已安装服务器依赖：`pip install 'glmocr[server]'`

**Q: 提交文档后一直显示 processing？**
A: 检查 Redis 是否正常运行，以及 GPU/CPU 资源是否充足。

**Q: WebSocket 连接失败？**
A: 确认服务器地址和端口正确，如果使用 HTTPS，WebSocket 应使用 `wss://` 协议。
