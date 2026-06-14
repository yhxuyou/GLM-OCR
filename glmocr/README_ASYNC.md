# GLM-OCR Async Service

基于 GLM-OCR SDK 的异步文档处理服务，支持实时进度推送。

## 特性

- **异步处理**：提交文档后立即返回，后台处理
- **实时进度**：通过 WebSocket 实时推送处理进度
- **Redis 支持**：使用 Redis 跟踪文档处理状态
- **向后兼容**：保持原有同步 API 不变

## 安装

```bash
# 安装基础依赖
pip install glmocr

# 安装异步服务依赖
pip install "glmocr[server]" redis fastapi uvicorn websockets
```

## 快速开始

### 1. 启动 Redis

```bash
# 使用 Docker 启动 Redis
docker run -d -p 6379:6379 redis:alpine

# 或使用本地 Redis
redis-server
```

### 2. 配置

创建 `config.yaml`：

```yaml
server:
  host: 0.0.0.0
  port: 8000

pipeline:
  redis:
    url: redis://localhost:6379/0
    key_prefix: glmocr
    max_connections: 10
  
  # 其他 pipeline 配置...
  maas:
    enabled: false
  
  ocr_api:
    api_host: localhost
    api_port: 8000
```

或通过环境变量：

```bash
export GLMOCR_REDIS_URL=redis://localhost:6379/0
```

### 3. 启动服务

```bash
# 使用默认配置
python -m glmocr.async_server

# 指定配置文件
python -m glmocr.async_server --config config.yaml

# 指定端口
python -m glmocr.async_server --port 8000
```

## API 文档

### POST /parse/async

提交文档进行异步处理。

**请求**：

```bash
curl -X POST http://localhost:8000/parse/async \
  -F "file=@document.pdf"
```

**响应**：

```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing"
}
```

### GET /parse/status/{doc_id}

查询文档处理进度。

**请求**：

```bash
curl http://localhost:8000/parse/status/550e8400-e29b-41d4-a716-446655440000
```

**响应**：

```json
{
  "completed": 5,
  "total": 10,
  "status": "processing"
}
```

状态值：
- `processing`：处理中
- `completed`：处理完成
- `not_found`：文档不存在

### GET /parse/result/{doc_id}

获取处理结果。

**请求**：

```bash
curl http://localhost:8000/parse/result/550e8400-e29b-41d4-a716-446655440000
```

**响应**（处理完成）：

```json
{
  "unit_0": {
    "json_result": [...],
    "markdown_result": "...",
    "original_images": [...]
  },
  "unit_1": {
    "json_result": [...],
    "markdown_result": "...",
    "original_images": [...]
  }
}
```

**响应**（处理中）：

```
HTTP 202 Accepted
{"detail": "Document processing not yet complete"}
```

### WebSocket /ws/{doc_id}

实时接收处理进度和结果。

**Python 示例**：

```python
import asyncio
import websockets
import json

async def monitor_progress(doc_id):
    uri = f"ws://localhost:8000/ws/{doc_id}"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "progress":
                print(f"进度: {data['completed']}/{data['total']}")
            
            elif data["type"] == "complete":
                print("处理完成！")
                result = data["result"]
                # 处理结果...
                break
            
            elif data["type"] == "error":
                print(f"错误: {data['message']}")
                break

asyncio.run(monitor_progress("550e8400-e29b-41d4-a716-446655440000"))
```

## 客户端示例

### Python 客户端

```python
import requests
import time
import json

# 提交文档
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/parse/async",
        files={"file": f}
    )
doc_id = response.json()["doc_id"]
print(f"文档 ID: {doc_id}")

# 轮询状态
while True:
    response = requests.get(f"http://localhost:8000/parse/status/{doc_id}")
    status = response.json()
    
    print(f"进度: {status['completed']}/{status['total']}")
    
    if status["status"] == "completed":
        break
    
    time.sleep(1)

# 获取结果
response = requests.get(f"http://localhost:8000/parse/result/{doc_id}")
result = response.json()

# 处理结果
for unit_id, unit_result in result.items():
    print(f"单元 {unit_id}:")
    print(f"  Markdown: {unit_result['markdown_result'][:100]}...")
```

### JavaScript 客户端

```javascript
// 提交文档
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/parse/async', {
  method: 'POST',
  body: formData
});

const { doc_id } = await response.json();

// 使用 WebSocket 监听进度
const ws = new WebSocket(`ws://localhost:8000/ws/${doc_id}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'progress') {
    console.log(`进度: ${data.completed}/${data.total}`);
    updateProgressBar(data.completed, data.total);
  }
  
  if (data.type === 'complete') {
    console.log('处理完成！', data.result);
    displayResults(data.result);
    ws.close();
  }
};
```

## 架构说明

```
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│  客户端      │────>│  Async Server │────>│  Pipeline │
│  (HTTP/WS)  │     │  (FastAPI)    │     │  (OCR)   │
└─────────────┘     └──────┬───────┘     └────┬─────┘
                           │                  │
                    ┌──────▼──────┐     ┌─────▼──────┐
                    │  Aggregator  │     │   vLLM     │
                    │  (Redis)     │<────│  (GPU)     │
                    └─────────────┘     └────────────┘
```

1. **Async Server**：接收请求，管理生命周期
2. **Pipeline**：执行文档处理（布局检测 + OCR）
3. **Aggregator**：使用 Redis 跟踪处理进度
4. **WebSocket**：实时推送进度更新

## 配置选项

### Redis 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `redis.url` | `redis://localhost:6379/0` | Redis 连接 URL |
| `redis.key_prefix` | `glmocr` | Redis key 前缀 |
| `redis.max_connections` | `10` | 最大连接数 |

### 环境变量

```bash
# Redis 配置
export GLMOCR_REDIS_URL=redis://localhost:6379/0

# 服务配置
export GLMOCR_SERVER_HOST=0.0.0.0
export GLMOCR_SERVER_PORT=8000

# 日志级别
export GLMOCR_LOG_LEVEL=INFO
```

## 与同步服务的对比

| 特性 | 同步服务 (server.py) | 异步服务 (async_server.py) |
|------|---------------------|---------------------------|
| 响应方式 | 阻塞等待结果 | 立即返回，后台处理 |
| 进度跟踪 | 无 | Redis + WebSocket |
| 适用场景 | 简单脚本、小批量 | 生产环境、大批量 |
| 依赖 | Flask | FastAPI + Redis |
| 实时反馈 | 无 | WebSocket 推送 |

## 故障排查

### Redis 连接失败

```
ERROR: RegionAggregator is not connected
```

**解决**：确保 Redis 服务正在运行

```bash
redis-cli ping  # 应返回 PONG
```

### WebSocket 连接失败

```
WebSocket connection failed
```

**解决**：检查防火墙设置，确保 WebSocket 端口未被阻止

### 处理超时

```
Document processing timeout
```

**解决**：检查 vLLM 服务状态，增加超时配置

## 许可证

与 GLM-OCR SDK 保持一致。
