# Medical OCR

专业的医疗文档 OCR 处理库，基于 glmocr 构建，保持原有的 Pipeline 推理逻辑不变，只替换了关键组件。

## 核心特点

✅ **不改变原有逻辑** - MedicalOcrPipeline 继承自 glmocr.Pipeline，所有 process() 方法完全继承  
✅ **替换关键组件** - 使用 MedicalPageLoader、MedicalLayoutDetector 和 MedicalResultFormatter  
✅ **三阶段预处理** - YOLO 检测 -> 方向矫正 -> 畸变矫正  
✅ **RapidOCR 坐标匹配** - 使用 RapidOCR 获取文字坐标并与 OCR 结果匹配  
✅ **可无缝替换** - 现有代码只需要替换 Pipeline 类即可

## 快速开始

### 安装依赖

```bash
pip install -e .
pip install rapidocr_onnxruntime  # 用于坐标检测
pip install ultralytics  # YOLO 文档检测 (可选)
pip install opencv-python
```

### 基本使用

与原始 glmocr.Pipeline 用法完全一致：

```python
from medical_ocr import MedicalOcrPipeline
from glmocr.config import load_config

config = load_config()

pipeline = MedicalOcrPipeline(
    config=config.pipeline,
    yolo_model_dir="/path/to/yolo/model",  # 可选
    uvdoc_model_dir="/path/to/uvdoc/model",  # 可选
)

pipeline.start()

for result in pipeline.process(request_data):
    print(result.json_result)
    print(result.markdown_result)

pipeline.stop()
```

## 核心类说明

### MedicalPageLoader
继承自 `glmocr.dataloader.PageLoader`，增加预处理功能：

- `detect_document(image)` - YOLO 文档检测
- `correct_orientation(image)` - RapidOCR 方向矫正
- `correct_distortion(image)` - UVDoc 畸变矫正

### MedicalLayoutDetector
继承自 `glmocr.layout.PPDocLayoutDetector`：

```
原始检测结果 → 过滤保留 "table" → 添加全图 "text"
```

### MedicalResultFormatter
继承自 `glmocr.postprocess.ResultFormatter`：

**核心功能：RapidOCR 坐标匹配**

```
OCR 文本内容
    ↓
RapidOCR 文字检测（获取坐标）
    ↓
坐标与文字匹配
    ↓
输出带坐标的 JSON 结果
```

#### 输出格式示例

```json
{
  "index": 0,
  "label": "text",
  "content": "Patient Name: John Doe",
  "score": 0.95,
  "bbox_2d": [100, 50, 400, 80],
  "polygon": [[100, 50], [400, 50], [400, 80], [100, 80]]
}
```

### MedicalOcrPipeline
继承自 `glmocr.pipeline.Pipeline`：

- **__init__ 唯一修改** - 替换 PageLoader、LayoutDetector 和 ResultFormatter
- **其他方法** - 完全继承自父类，无需修改

## 项目结构

```
medical_ocr/
├── medical_ocr/
│   ├── __init__.py              # 包入口
│   ├── page_loader.py           # MedicalPageLoader (图片预处理)
│   ├── layout_detector.py       # MedicalLayoutDetector (布局过滤)
│   ├── result_formatter.py      # MedicalResultFormatter (坐标匹配)
│   ├── pipeline.py              # MedicalOcrPipeline (继承自 glmocr.Pipeline)
│   └── uvdoc_inference.py       # UVDoc 推理包装
├── examples/
│   └── basic_usage.py           # 使用示例
├── tests/
├── pyproject.toml
└── README.md
```

## 迁移指南

### 原代码 (glmocr)
```python
from glmocr.pipeline import Pipeline

config = load_config()
pipeline = Pipeline(config.pipeline)
pipeline.start()
for result in pipeline.process(request_data):
    print(result.json_result)
pipeline.stop()
```

### 新代码 (medical-ocr)
```python
from medical_ocr import MedicalOcrPipeline  # 只需改这一行

config = load_config()
pipeline = MedicalOcrPipeline(
    config=config.pipeline,
    yolo_model_dir="/path/to/yolo",  # 可选
    uvdoc_model_dir="/path/to/uvdoc",  # 可选
)
pipeline.start()
for result in pipeline.process(request_data):
    print(result.json_result)  # JSON 现在包含坐标信息！
pipeline.stop()
```

## 高并发服务器

### 快速启动

#### 开发模式

```bash
# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
python -m medical_ocr.high_perf_server
```

#### 生产模式

```bash
# 使用启动脚本
./start_server.sh prod

# 或者使用 uvicorn 直接启动
uvicorn medical_ocr.high_perf_server:app --host 0.0.0.0 --port 8080 --workers 4 --loop uvloop
```

#### Docker 部署

```bash
# 使用 Docker Compose
docker-compose up -d

# 检查状态
docker-compose ps

# 查看日志
docker-compose logs -f medical-ocr
```

### API 接口

#### 1. 健康检查

```bash
curl http://localhost:8080/health
```

#### 2. 单文档处理

```bash
curl -X POST http://localhost:8080/ocr/parse \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["https://example.com/medical_doc.jpg"]
  }'
```

#### 3. 批量处理

```bash
curl -X POST http://localhost:8080/ocr/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"images": ["https://example.com/doc1.jpg"]},
      {"images": ["https://example.com/doc2.jpg"]}
    ]
  }'
```

#### 4. 缓存管理

```bash
# 查看缓存统计
curl http://localhost:8080/cache/stats

# 清空缓存
curl -X DELETE http://localhost:8080/cache
```

#### 5. 监控指标

```bash
# Prometheus metrics
curl http://localhost:8001/metrics
```

### 高并发特性

| 特性 | 说明 |
|------|------|
| **异步处理** | 使用 FastAPI + asyncio 实现非阻塞请求处理 |
| **线程池** | 线程池执行 CPU 密集型 OCR 任务 |
| **Redis 缓存** | 自动缓存重复请求，支持 TTL 过期 |
| **批量处理** | 支持最多 50 个请求批量处理 |
| **监控指标** | Prometheus 指标监控 |
| **优雅关闭** | 支持信号处理和资源清理 |

### 配置说明

编辑 `server_config.ini` 或通过环境变量配置：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| REDIS_HOST | localhost | Redis 主机 |
| REDIS_PORT | 6379 | Redis 端口 |
| THREAD_POOL_SIZE | 8 | 线程池大小 |
| PROMETHEUS_PORT | 8001 | 监控端口 |
| MAX_BATCH_SIZE | 50 | 批量处理上限 |

## 许可证

Apache-2.0
