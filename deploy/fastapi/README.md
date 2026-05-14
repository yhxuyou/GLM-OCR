# GLM-OCR FastAPI 高并发部署方案

这是一个专为GLM-OCR设计的FastAPI高并发部署方案，支持异步任务处理、请求队列、速率限制等特性。

## 特性

- 🚀 **高并发处理**: 基于异步任务队列的并发处理
- 📋 **任务管理**: 完整的任务状态追踪和管理
- ⏱️ **速率限制**: 可配置的客户端请求限流
- 🔄 **同步/异步双模式**: 支持直接返回和异步任务两种方式
- 📊 **实时监控**: 健康检查和统计信息API
- 🐳 **Docker支持**: 完整的Docker和docker-compose部署方案
- 🛡️ **优雅降级**: 队列满时自动拒绝请求

## 目录结构

```
deploy/fastapi/
├── __init__.py              # 包初始化
├── config.py                # 配置管理
├── logger.py                # 日志模块
├── task_manager.py          # 任务管理器（核心）
├── ocr_processor.py         # GLM-OCR集成
├── main.py                  # FastAPI应用入口
├── requirements.txt         # Python依赖
├── Dockerfile              # Docker镜像构建
├── docker-compose.yml      # Docker Compose配置
├── .env.example            # 环境变量示例
├── start.sh                # 快速启动脚本
├── test_client.py          # 测试客户端
└── README.md               # 本文件
```

## 快速开始

### 方法1: 使用启动脚本（推荐用于开发）

```bash
# 进入部署目录
cd deploy/fastapi

# 复制环境变量配置
cp .env.example .env

# 编辑配置（根据需要修改）
# nano .env

# 使用启动脚本
chmod +x start.sh
./start.sh
```

### 方法2: 手动启动

```bash
# 1. 安装依赖
pip install -e ".[all]"
pip install -r deploy/fastapi/requirements.txt

# 2. 配置环境变量（可选）
cp deploy/fastapi/.env.example .env
# 编辑 .env 文件

# 3. 启动服务
uvicorn deploy.fastapi.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
```

### 方法3: 使用Docker

```bash
# 进入部署目录
cd deploy/fastapi

# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## API文档

服务启动后，可以访问以下地址：

- 服务地址: http://localhost:8000
- Swagger文档: http://localhost:8000/docs
- ReDoc文档: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/health

## API端点

### 1. 健康检查

```bash
GET /health
```

响应示例:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "queue_stats": {...},
  "ocr_stats": {...}
}
```

### 2. 同步OCR（不推荐高并发）

```bash
POST /api/v1/ocr/sync
Content-Type: multipart/form-data

file: [图片或PDF文件]
save_layout_visualization: false (可选)
```

响应示例:
```json
{
  "success": true,
  "result": {
    "json_result": [...],
    "markdown_result": "...",
    "original_images": [...]
  }
}
```

### 3. 异步OCR（推荐用于高并发）

```bash
POST /api/v1/ocr/async
Content-Type: multipart/form-data

file: [图片或PDF文件]
save_layout_visualization: false (可选)
```

响应示例:
```json
{
  "success": true,
  "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 4. 查询任务状态

```bash
GET /api/v1/tasks/{task_id}
```

响应示例:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {...},
  "error": null
}
```

### 5. 取消任务

```bash
POST /api/v1/tasks/{task_id}/cancel
```

### 6. 列出所有任务

```bash
GET /api/v1/tasks?status=pending
```

### 7. 获取统计信息

```bash
GET /api/v1/stats
```

## 使用示例

### Python客户端

```python
# 使用test_client.py
python deploy/fastapi/test_client.py \
    --file /path/to/image.jpg \
    --mode async

# 或者使用requests
import requests

# 异步OCR
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/ocr/async',
        files={'file': f}
    )
task_id = response.json()['task_id']

# 查询结果
result = requests.get(f'http://localhost:8000/api/v1/tasks/{task_id}').json()
if result['status'] == 'completed':
    print(result['result']['markdown_result'])
```

### cURL示例

```bash
# 异步上传
curl -X POST http://localhost:8000/api/v1/ocr/async \
  -F "file=@/path/to/image.jpg"

# 查询任务状态
curl http://localhost:8000/api/v1/tasks/{task_id}
```

## 配置说明

主要配置项（通过环境变量或.env文件）:

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| HOST | 监听地址 | 0.0.0.0 |
| PORT | 监听端口 | 8000 |
| DEBUG | 调试模式 | false |
| LOG_LEVEL | 日志级别 | INFO |
| MAX_UPLOAD_SIZE | 最大文件大小（字节） | 52428800 (50MB) |
| MAX_QUEUE_SIZE | 任务队列最大长度 | 1000 |
| MAX_CONCURRENT_TASKS | 最大并发任务数 | 4 |
| TASK_TIMEOUT | 任务超时时间（秒） | 300 |
| RATE_LIMIT_ENABLED | 是否启用速率限制 | true |
| RATE_LIMIT_REQUESTS | 时间窗口内最大请求数 | 100 |
| RATE_LIMIT_WINDOW | 速率限制时间窗口（秒） | 60 |
| GLMOCR_MODE | GLM-OCR模式 | selfhosted |
| CUDA_VISIBLE_DEVICES | GPU设备 | 未设置 |

## 性能调优建议

### 1. 并发数设置

根据GPU显存和性能调整:
- 单个GPU: `MAX_CONCURRENT_TASKS=2-4`
- 多GPU: 可适当增加，但不建议超过GPU数量×2

### 2. 队列大小

根据预期并发量调整:
- 低并发: `MAX_QUEUE_SIZE=100`
- 中并发: `MAX_QUEUE_SIZE=1000`
- 高并发: `MAX_QUEUE_SIZE=5000+`

### 3. Uvicorn Worker

**注意**: 建议使用单个worker (`--workers 1`)，因为:
- GLM-OCR内部已有并发处理
- 多worker会导致模型加载多份，占用大量显存
- 任务队列已在单个进程内管理

### 4. GPU优化

确保正确配置:
```bash
# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=0

# 使用docker-compose时取消注释GPU配置
```

## 任务状态说明

| 状态 | 说明 |
|------|------|
| pending | 任务在队列中等待 |
| processing | 任务正在处理 |
| completed | 任务成功完成 |
| failed | 任务处理失败 |
| cancelled | 任务被取消 |
| timeout | 任务超时 |

## 监控和日志

### 日志查看

```bash
# Docker方式
docker-compose logs -f

# 直接运行时
# 日志输出到控制台
```

### 健康检查

```bash
# 定期检查服务状态
curl http://localhost:8000/health
```

### 统计信息

```bash
# 获取队列和OCR统计
curl http://localhost:8000/api/v1/stats
```

## 故障排查

### 常见问题

1. **模型加载失败**
   - 检查模型路径配置
   - 确认GPU显存足够
   - 查看日志中的错误信息

2. **任务超时**
   - 增加`TASK_TIMEOUT`配置
   - 检查文件大小是否过大
   - 确认GPU是否正常工作

3. **队列满**
   - 增加`MAX_QUEUE_SIZE`
   - 增加`MAX_CONCURRENT_TASKS`
   - 考虑扩展部署

4. **速率限制**
   - 调整`RATE_LIMIT_REQUESTS`
   - 或设置`RATE_LIMIT_ENABLED=false`禁用

## 生产部署建议

1. **使用反向代理**: Nginx或Traefik
2. **HTTPS**: 配置SSL证书
3. **监控**: 集成Prometheus + Grafana
4. **日志**: 使用ELK或Loki收集日志
5. **备份**: 定期备份重要数据
6. **高可用**: 考虑多实例部署（注意模型显存）

## 许可证

与GLM-OCR项目使用相同的许可证。
