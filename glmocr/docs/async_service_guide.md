# GLM-OCR 异步服务文档

## 目录

- [一、启动方式](#一启动方式)
  - [1.1 前置依赖](#11-前置依赖)
  - [1.2 启动 Redis](#12-启动-redis)
  - [1.3 配置文件](#13-配置文件)
  - [1.4 启动服务](#14-启动服务)
  - [1.5 验证启动](#15-验证启动)
- [二、数据流程调用说明](#二数据流程调用说明)
  - [2.1 整体架构图](#21-整体架构图)
  - [2.2 详细数据流程](#22-详细数据流程)
- [三、20 个并发调用的数据处理](#三20-个并发调用的数据处理)
  - [3.1 并发场景描述](#31-并发场景描述)
  - [3.2 并发处理流程](#32-并发处理流程)
  - [3.3 并发控制机制](#33-并发控制机制)
  - [3.4 并发性能指标](#34-并发性能指标)
  - [3.5 并发场景下的资源消耗](#35-并发场景下的资源消耗)
  - [3.6 调优建议](#36-调优建议)
- [四、预处理与后处理服务](#四预处理与后处理服务)
  - [4.1 预处理服务](#41-预处理服务)
  - [4.2 后处理服务](#42-后处理服务)
  - [4.3 完整部署架构](#43-完整部署架构)
- [五、API 参考](#五api-参考)
- [六、故障排查](#六故障排查)
- [七、配置参考](#七配置参考)

---

## 一、启动方式

### 1.1 前置依赖

```bash
# 安装基础依赖
pip install glmocr

# 安装异步服务依赖
pip install "glmocr[server]" redis fastapi uvicorn websockets httpx
```

### 1.2 启动 Redis

```bash
# 方式一：Docker（推荐）
docker run -d -p 6379:6379 redis:alpine

# 方式二：本地安装
redis-server
```

### 1.3 配置文件

创建 `config.yaml`：

```yaml
server:
  host: 0.0.0.0
  port: 8000

pipeline:
  # Redis 配置（用于结果聚合）
  redis:
    url: redis://localhost:6379/0
    key_prefix: glmocr
    max_connections: 10
  
  # MaaS 模式（云端 API）
  maas:
    enabled: true
    api_key: ${ZHIPU_API_KEY}
  
  # 自部署 OCR API 配置
  ocr_api:
    api_host: localhost
    api_port: 5002
    api_path: /v1/chat/completions
    connection_pool_size: 128
  
  # 异步 OCR 客户端配置
  async_ocr:
    max_connections: 100        # httpx 连接池大小
    max_concurrent_requests: 50 # 最大并发请求数
  
  # 异步 Pipeline 配置
  async_pipeline:
    max_concurrent_regions: 100  # 最大并发 region 数
    enable_batch_processing: true
  
  # 布局检测配置
  layout:
    device: cuda:0
    batch_size: 8
```

### 1.4 启动服务

```bash
# 方式一：使用配置文件
python -m glmocr.async_server --config config.yaml

# 方式二：使用环境变量
export GLMOCR_REDIS_URL=redis://localhost:6379/0
export GLMOCR_OCR_API_HOST=localhost
export GLMOCR_OCR_API_PORT=5002
python -m glmocr.async_server --port 8000

# 方式三：指定日志级别
python -m glmocr.async_server --config config.yaml --log-level DEBUG
```

### 1.5 验证启动

```bash
# 健康检查
curl http://localhost:8000/health
# 返回: {"status": "ok"}
```

---

## 二、数据流程调用说明

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              客户端层                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │
│  │ 提交文档  │  │ 查询状态  │  │ 获取结果  │  │ WebSocket │                    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                    │
└───────┼──────────────┼──────────────┼──────────────┼────────────────────────┘
        │              │              │              │
┌───────▼──────────────▼──────────────▼──────────────▼────────────────────────┐
│                         Async Server (FastAPI)                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  /parse/async  →  创建后台任务  →  返回 doc_id                        │   │
│  │  /parse/status →  查询 Redis   →  返回进度                            │   │
│  │  /parse/result →  查询 Redis   →  返回结果                            │   │
│  │  /ws/{doc_id}  →  轮询 Redis   →  推送进度/结果                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │      AsyncPipeline            │
                    │  ┌─────────────────────────┐  │
                    │  │ 1. 加载页面 (线程池)     │  │
                    │  │ 2. 布局检测 (线程池)     │  │
                    │  │ 3. 提交 region (异步)    │  │
                    │  │ 4. 立即返回 doc_id       │  │
                    │  └─────────────────────────┘  │
                    └───────────────┬───────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
    ┌─────────▼─────────┐ ┌────────▼────────┐ ┌─────────▼─────────┐
    │  Region 0 处理     │ │  Region 1 处理   │ │  Region N 处理     │
    │  ┌──────────────┐ │ │  ┌─────────────┐ │ │  ┌──────────────┐ │
    │  │ 裁剪图片      │ │ │  │ 裁剪图片     │ │ │  │ 裁剪图片      │ │
    │  │ 构建请求      │ │ │  │ 构建请求     │ │ │  │ 构建请求      │ │
    │  │ 异步OCR请求   │ │ │  │ 异步OCR请求  │ │ │  │ 异步OCR请求   │ │
    │  └──────┬───────┘ │ │  └──────┬──────┘ │ │  └──────┬───────┘ │
    └─────────┼─────────┘ └─────────┼────────┘ └─────────┼─────────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │     AsyncOCRClient            │
                    │  ┌─────────────────────────┐  │
                    │  │  httpx.AsyncClient      │  │
                    │  │  - 连接池管理            │  │
                    │  │  - 异步请求              │  │
                    │  │  - 重试机制              │  │
                    │  └─────────────────────────┘  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │         vLLM / OCR API        │
                    │      (GPU 推理服务)            │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │     RegionAggregator          │
                    │  ┌─────────────────────────┐  │
                    │  │  Redis Hash             │  │
                    │  │  - doc:{id}:meta        │  │
                    │  │  - doc:{id}:regions     │  │
                    │  │  - 原子计数              │  │
                    │  └─────────────────────────┘  │
                    └───────────────────────────────┘
```

### 2.2 详细数据流程

#### 步骤 1：提交文档

```
客户端 → POST /parse/async → Async Server
```

1. 客户端上传文件（multipart/form-data）
2. 服务端生成唯一 `doc_id`（UUID）
3. 在 Redis 中注册文档：`doc:{doc_id}:meta = {total: 1, completed: 0, status: processing}`
4. 创建后台任务 `asyncio.create_task(_process_document_background())`
5. **立即返回** `{"doc_id": "xxx", "status": "processing"}`

#### 步骤 2：后台处理

```
后台任务 → AsyncPipeline.process_async()
```

1. **加载页面**（线程池执行，I/O 密集）
   - 读取临时文件
   - 解析 PDF/图片
   - 返回页面列表

2. **布局检测**（线程池执行，CPU 密集）
   - 对每个页面运行布局检测模型
   - 返回所有 region 的 bbox 和类型

3. **注册文档**
   - 更新 Redis：`doc:{doc_id}:meta.total = N`（N = region 数量）

4. **提交所有 region**（异步，fire-and-forget）
   - 为每个 region 创建 `asyncio.Task`
   - 使用 `Semaphore` 控制并发数（默认 100）
   - **立即返回 doc_id**，不等待 OCR 结果

#### 步骤 3：Region 异步处理

```
每个 region task → AsyncOCRClient → vLLM → RegionAggregator
```

对每个 region 并行执行：

1. **裁剪图片**
   - 根据 bbox 裁剪原图

2. **构建请求**
   - 将裁剪后的图片编码为 base64
   - 构建 OpenAI 格式的请求

3. **异步 OCR 请求**
   - 通过 `httpx.AsyncClient` 发送请求
   - 支持重试（指数退避）
   - 等待响应

4. **记录结果**
   - 调用 `aggregator.on_region_complete(doc_id, region_id, result)`
   - Redis 原子操作：
     - `HSET doc:{doc_id}:regions {region_id} {result_json}`
     - `HINCRBY doc:{doc_id}:meta completed 1`
   - 检查是否全部完成：`completed == total`
   - 如果全部完成，更新状态：`status = complete`

#### 步骤 4：查询进度/获取结果

```
客户端 → GET /parse/status/{doc_id} → Redis → 返回进度
客户端 → GET /parse/result/{doc_id} → Redis → 返回结果
客户端 → WebSocket /ws/{doc_id} → 轮询 Redis → 推送进度
```

**查询进度**：
- 读取 Redis：`HMGET doc:{doc_id}:meta completed total status`
- 返回：`{"completed": 5, "total": 10, "status": "processing"}`

**获取结果**：
- 检查是否完成：`status == "complete"`
- 如果未完成，返回 HTTP 202
- 如果完成，读取所有 region 结果：`HGETALL doc:{doc_id}:regions`
- 返回完整结果

**WebSocket 推送**：
- 每 1.5 秒轮询 Redis
- 推送进度：`{"type": "progress", "completed": 5, "total": 10}`
- 全部完成时推送：`{"type": "complete", "result": {...}}`

---

## 三、20 个并发调用的数据处理

### 3.1 并发场景描述

假设 20 个客户端同时调用 `POST /parse/async`，每个文档包含 10 个 region。

### 3.2 并发处理流程

```
时间轴 ──────────────────────────────────────────────────────────────────────►

T0: 20 个请求同时到达
    │
    ├─ 请求 1  ──→ 生成 doc_1  ──→ 注册 Redis  ──→ 创建后台任务  ──→ 返回 doc_1
    ├─ 请求 2  ──→ 生成 doc_2  ──→ 注册 Redis  ──→ 创建后台任务  ──→ 返回 doc_2
    ├─ 请求 3  ──→ 生成 doc_3  ──→ 注册 Redis  ──→ 创建后台任务  ──→ 返回 doc_3
    │   ...
    └─ 请求 20 ──→ 生成 doc_20 ──→ 注册 Redis  ──→ 创建后台任务  ──→ 返回 doc_20

T1: 20 个后台任务开始执行（并发）
    │
    ├─ 任务 1  ──→ 加载页面 ──→ 布局检测 ──→ 发现 10 个 region ──→ 注册 total=10
    ├─ 任务 2  ──→ 加载页面 ──→ 布局检测 ──→ 发现 8 个 region  ──→ 注册 total=8
    │   ...
    └─ 任务 20 ──→ 加载页面 ──→ 布局检测 ──→ 发现 12 个 region ──→ 注册 total=12

T2: 所有 region 开始异步提交（受 Semaphore 控制）
    │
    │  总 region 数 = 10 + 8 + ... + 12 = 200 个 region
    │  Semaphore 限制 = 100（可配置）
    │
    │  第一批：100 个 region 同时提交到 vLLM
    │  ├─ doc_1:region_0 ──→ OCR 请求
    │  ├─ doc_1:region_1 ──→ OCR 请求
    │  ├─ doc_2:region_0 ──→ OCR 请求
    │  │   ...
    │  └─ doc_10:region_5 ──→ OCR 请求
    │
    │  第二批：等待第一批完成后，剩余 100 个 region 提交
    │  ├─ doc_10:region_6 ──→ OCR 请求
    │  │   ...
    │  └─ doc_20:region_11 ──→ OCR 请求

T3: vLLM 返回结果（异步，乱序）
    │
    │  vLLM 可能同时处理多个请求，返回顺序不确定
    │
    ├─ doc_1:region_0 完成 ──→ Redis HINCRBY completed 1 ──→ 检查: 1/10
    ├─ doc_3:region_2 完成 ──→ Redis HINCRBY completed 1 ──→ 检查: 1/15
    ├─ doc_1:region_1 完成 ──→ Redis HINCRBY completed 1 ──→ 检查: 2/10
    │   ...
    └─ doc_20:region_11 完成 ──→ Redis HINCRBY completed 1 ──→ 检查: 12/12 ✓ 完成!

T4: 客户端查询结果
    │
    ├─ 客户端 1  ──→ GET /parse/status/doc_1 ──→ {"completed": 5, "total": 10, "status": "processing"}
    ├─ 客户端 20 ──→ GET /parse/status/doc_20 ──→ {"completed": 12, "total": 12, "status": "completed"}
    └─ 客户端 20 ──→ GET /parse/result/doc_20 ──→ 返回完整结果
```

### 3.3 并发控制机制

#### 3.3.1 Semaphore 控制 Region 并发

```python
# AsyncPipeline 中的 Semaphore
self._region_semaphore = asyncio.Semaphore(100)  # 默认 100

async def _submit_region_async(self, ...):
    async with self._region_semaphore:  # 限制同时处理的 region 数
        # 裁剪图片
        # 构建请求
        # 异步 OCR 请求
        # 记录结果
```

**作用**：
- 防止同时向 vLLM 发送过多请求
- 避免 vLLM 过载或 OOM
- 可根据 vLLM 的 GPU 显存和吞吐能力调整

#### 3.3.2 httpx 连接池控制

```python
# AsyncOCRClient 中的连接池
self._client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=128,           # 最大连接数
        max_keepalive_connections=128, # 保持活动的连接数
    ),
    timeout=httpx.Timeout(120),
)
```

**作用**：
- 复用 TCP 连接，减少连接建立开销
- 限制并发连接数，防止资源耗尽

#### 3.3.3 Redis 原子操作

```python
# RegionAggregator 中的原子操作
async with r.pipeline(transaction=True) as pipe:
    pipe.hset(regions_key, region_id, serialized)  # 存储结果
    pipe.hincrby(meta_key, "completed", 1)          # 原子递增计数
    await pipe.execute()
```

**作用**：
- 防止并发写入导致计数错误
- 确保数据一致性

### 3.4 并发性能指标

| 指标 | 默认值 | 说明 |
|------|--------|------|
| 最大并发文档数 | 无限制 | 取决于内存和 Redis 容量 |
| 最大并发 region 数 | 100 | `async_pipeline.max_concurrent_regions` |
| httpx 最大连接数 | 128 | `async_ocr.max_connections` |
| Redis 连接池 | 10 | `redis.max_connections` |

### 3.5 并发场景下的资源消耗

假设 20 个并发文档，每个文档 10 个 region：

| 资源 | 消耗 | 说明 |
|------|------|------|
| 内存 | ~2GB | 20 个文档的图片数据 |
| Redis | ~200 keys | 20 个文档 × 10 个 region |
| vLLM 并发 | 100 | Semaphore 限制 |
| httpx 连接 | ~100 | 活跃连接数 |

### 3.6 调优建议

#### 提高吞吐量

```yaml
pipeline:
  async_ocr:
    max_connections: 200        # 增加连接池
    max_concurrent_requests: 100
  
  async_pipeline:
    max_concurrent_regions: 200  # 增加并发 region 数
```

#### 降低资源消耗

```yaml
pipeline:
  async_ocr:
    max_connections: 50         # 减少连接池
    max_concurrent_requests: 20
  
  async_pipeline:
    max_concurrent_regions: 50   # 减少并发 region 数
```

#### 适配 vLLM 能力

```yaml
# 如果 vLLM 每秒处理 10 个 region
pipeline:
  async_pipeline:
    max_concurrent_regions: 10   # 匹配 vLLM 吞吐

# 如果 vLLM 有 4 张 GPU
pipeline:
  async_ocr:
    max_connections: 40          # 每张 GPU 10 个并发
```

---

## 四、预处理与后处理服务

### 4.1 预处理服务

预处理服务提供文档图像的预处理能力，包括文档检测、方向矫正和扭曲矫正。

#### 4.1.1 启动预处理服务

```bash
# 方式一：直接启动
python -m preprocess.server --config preprocess/config.yaml

# 方式二：使用环境变量
export PREPROCESS_DEVICE=cuda:0
export PREPROCESS_PORT=5003
python -m preprocess.server
```

#### 4.1.2 预处理 API

**单步骤调用**：

```bash
# 文档检测
curl -X POST http://localhost:5003/detect \
  -F "file=@document.jpg"

# 方向矫正
curl -X POST http://localhost:5003/orient \
  -F "file=@document.jpg" \
  -F "apply_rotation=true"

# 扭曲矫正
curl -X POST http://localhost:5003/dewarp \
  -F "file=@document.jpg"
```

**完整流水线**：

```bash
# 串行执行：文档检测 → 方向矫正 → 扭曲矫正
curl -X POST http://localhost:5003/preprocess \
  -F "file=@document.jpg" \
  -F "return_intermediate=true"
```

**响应示例**：

```json
{
  "original_size": [1920, 1080],
  "final_size": [1800, 900],
  "final_image": "base64_encoded_image...",
  "steps": [
    {
      "step": "doc_detection",
      "result": {
        "bbox": [100, 100, 1820, 980],
        "confidence": 0.95
      }
    },
    {
      "step": "orientation_correction",
      "result": {
        "angle": 0,
        "confidence": 0.99
      }
    },
    {
      "step": "dewarp_correction",
      "result": {
        "original_size": [1720, 880],
        "corrected_size": [1800, 900]
      }
    }
  ]
}
```

#### 4.1.3 GPU 模型部署

预处理服务支持多种 GPU 模型格式：

**PyTorch 模型**：
```python
# preprocess/models/doc_detector.py
self._model = torch.load(f"{model_dir}/model.pth", map_location=self._device)
self._model.eval()
```

**Hugging Face 模型**：
```python
from transformers import AutoModel
self._model = AutoModel.from_pretrained(model_dir).to(self._device)
```

**ONNX 模型**：
```python
import onnxruntime as ort
self._model = ort.InferenceSession(f"{model_dir}/model.onnx")
```

**TensorRT 模型**：
```python
import tensorrt as trt
# TRT 加载逻辑
```

### 4.2 后处理服务

后处理服务对 OCR 结果进行格式化和优化。

#### 4.2.1 启动后处理服务

```bash
python -m postprocess.server --config postprocess/config.yaml
```

#### 4.2.2 后处理 API

```bash
curl -X POST http://localhost:5004/postprocess \
  -H "Content-Type: application/json" \
  -d '{
    "text": "识别的文本内容...",
    "format": "markdown",
    "options": {
      "remove_extra_whitespace": true,
      "normalize_unicode": true
    }
  }'
```

**响应示例**：

```json
{
  "original_text": "识别的文本内容...",
  "processed_text": "格式化后的文本...",
  "format": "markdown",
  "metadata": {
    "processing_time_ms": 15,
    "operations_applied": ["remove_whitespace", "normalize_unicode"]
  }
}
```

### 4.3 完整部署架构

#### 4.3.1 独立部署模式

```
┌─────────────────┐
│  预处理服务      │  GPU 0
│  :5003          │
└────────┬────────┘
         │
┌────────▼────────┐
│  OCR 服务        │  GPU 1
│  :8000          │
└────────┬────────┘
         │
┌────────▼────────┐
│  后处理服务      │  CPU
│  :5004          │
└─────────────────┘
```

**启动命令**：

```bash
# 终端 1：启动预处理服务
python -m preprocess.server --port 5003

# 终端 2：启动 OCR 服务
python -m glmocr.async_server --port 8000

# 终端 3：启动后处理服务
python -m postprocess.server --port 5004
```

#### 4.3.2 Docker Compose 部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  preprocess:
    build: ./preprocess
    ports:
      - "5003:5003"
    environment:
      - PREPROCESS_DEVICE=cuda:0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
  
  ocr-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GLMOCR_REDIS_URL=redis://redis:6379/0
      - GLMOCR_OCR_API_HOST=vllm
      - GLMOCR_OCR_API_PORT=5002
    depends_on:
      - redis
      - preprocess
  
  postprocess:
    build: ./postprocess
    ports:
      - "5004:5004"
    depends_on:
      - ocr-service
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

**启动命令**：

```bash
docker-compose up -d
```

#### 4.3.3 集成到 AsyncPipeline

在 `config.yaml` 中启用预处理和后处理：

```yaml
pipeline:
  # 预处理配置
  preprocess:
    enabled: true
    service_url: http://localhost:5003
    timeout: 30
  
  # 后处理配置
  postprocess:
    enabled: true
    service_url: http://localhost:5004
    timeout: 10
```

**数据流程**：

```
原始图像 → 预处理服务 → OCR 服务 → 后处理服务 → 最终结果
   │           │            │            │
   │      文档检测       布局检测      文本格式化
   │      方向矫正       文字识别      结构优化
   │      扭曲矫正       表格识别
```

---

## 五、API 参考

### 4.1 POST /parse/async

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

### 4.2 GET /parse/status/{doc_id}

查询处理进度。

**请求**：
```bash
curl http://localhost:8000/parse/status/550e8400-e29b-41d4-a716-446655440000
```

**响应**：
```json
{
  "completed": 5,
  "total": 10,
  "status": "processing"  // processing | completed | not_found
}
```

### 4.3 GET /parse/result/{doc_id}

获取处理结果。

**请求**：
```bash
curl http://localhost:8000/parse/result/550e8400-e29b-41d4-a716-446655440000
```

**响应（完成）**：
```json
{
  "region_0": {
    "page_idx": 0,
    "region": {"bbox_2d": [...], "label": "text"},
    "content": "识别的文本内容"
  },
  "region_1": {
    "page_idx": 0,
    "region": {"bbox_2d": [...], "label": "table"},
    "content": "表格内容"
  }
}
```

**响应（未完成）**：
```
HTTP 202 Accepted
{"detail": "Document processing not yet complete"}
```

### 4.4 WebSocket /ws/{doc_id}

实时接收进度更新。

**Python 示例**：
```python
import asyncio
import websockets
import json

async def monitor(doc_id):
    async with websockets.connect(f"ws://localhost:8000/ws/{doc_id}") as ws:
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "progress":
                print(f"进度: {msg['completed']}/{msg['total']}")
            elif msg["type"] == "complete":
                print("完成!", msg["result"])
                break

asyncio.run(monitor("550e8400-e29b-41d4-a716-446655440000"))
```

---

## 五、故障排查

### 5.1 Redis 连接失败

```
ERROR: RegionAggregator is not connected
```

**解决**：
```bash
redis-cli ping  # 应返回 PONG
```

### 5.2 vLLM 过载

```
WARNING: OCR API request error: Connection pool is full
```

**解决**：
- 减少 `max_concurrent_regions`
- 增加 vLLM 实例数量
- 增加 `max_connections`

### 5.3 内存不足

```
ERROR: Background processing failed: MemoryError
```

**解决**：
- 减少并发文档数
- 限制 PDF 页数：`pdf_max_pages: 10`
- 降低图片分辨率

---

## 六、配置参考

### 6.1 完整配置示例

```yaml
server:
  host: 0.0.0.0
  port: 8000

pipeline:
  # Redis 配置
  redis:
    url: redis://localhost:6379/0
    key_prefix: glmocr
    max_connections: 10
  
  # MaaS 模式
  maas:
    enabled: false
  
  # OCR API 配置
  ocr_api:
    api_host: localhost
    api_port: 5002
    api_path: /v1/chat/completions
    api_key: null
    model: glm-ocr
    verify_ssl: false
    connect_timeout: 30
    request_timeout: 120
    retry_max_attempts: 2
    connection_pool_size: 128
  
  # 异步 OCR 配置
  async_ocr:
    max_connections: 100
    max_concurrent_requests: 50
  
  # 异步 Pipeline 配置
  async_pipeline:
    max_concurrent_regions: 100
    enable_batch_processing: true
  
  # 布局检测配置
  layout:
    model_dir: null
    device: cuda:0
    batch_size: 8
    threshold: 0.3
  
  # 页面加载配置
  page_loader:
    pdf_dpi: 200
    pdf_max_pages: null
    max_tokens: 8192
  
  # 结果格式化配置
  result_formatter:
    filter_nested: true
    output_format: both
```

### 6.2 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `GLMOCR_REDIS_URL` | Redis 连接 URL | `redis://localhost:6379/0` |
| `GLMOCR_OCR_API_HOST` | OCR API 主机 | `localhost` |
| `GLMOCR_OCR_API_PORT` | OCR API 端口 | `5002` |
| `GLMOCR_ASYNC_OCR_MAX_CONNECTIONS` | httpx 最大连接数 | `100` |
| `GLMOCR_ASYNC_PIPELINE_MAX_CONCURRENT_REGIONS` | 最大并发 region 数 | `100` |
| `ZHIPU_API_KEY` | API Key | - |
