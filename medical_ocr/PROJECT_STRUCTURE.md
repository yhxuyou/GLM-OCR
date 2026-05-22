# Medical OCR - Project Structure

## 📁 目录结构

```
medical_ocr/
├── medical_ocr/                    # 核心包
│   ├── __init__.py               # 包入口，导出公共 API
│   ├── constants.py              # 全局常量定义
│   │
│   ├── core/                      # 核心组件（继承自 glmocr）
│   │   ├── page_loader.py        # MedicalPageLoader
│   │   ├── layout_detector.py    # MedicalLayoutDetector
│   │   ├── result_formatter.py   # MedicalResultFormatter
│   │   └── pipeline.py           # MedicalOcrPipeline
│   │
│   ├── utils/                     # 工具模块
│   │   ├── __init__.py
│   │   ├── cache.py              # 缓存实现（SimpleCache/RedisCache）
│   │   ├── config.py             # 配置加载
│   │   └── logger.py             # 日志工具
│   │
│   ├── high_perf/                 # 高性能模块
│   │   ├── pipeline_pool.py      # Pipeline 池管理器
│   │   └── high_perf_server.py   # FastAPI 高性能服务器
│   │
│   └── models/                    # 模型相关
│       └── uvdoc_inference.py    # UVDoc 模型推理
│
├── examples/                      # 使用示例
│   ├── basic_usage.py            # 基础用法
│   └── high_perf_usage.py        # 高性能用法
│
├── tests/                         # 测试
│   └── __init__.py
│
├── config.yaml                   # 完整配置
├── config_simple.yaml             # 简化配置
├── pyproject.toml                # 项目配置
├── requirements.txt               # 依赖列表
└── README.md                     # 项目文档
```

## 🎯 核心组件

### 1. Core（核心组件）

| 文件 | 类 | 继承自 | 功能 |
|------|------|--------|------|
| `page_loader.py` | `MedicalPageLoader` | `PageLoader` | 图像预处理（YOLO + RapidOCR + UVDoc）|
| `layout_detector.py` | `MedicalLayoutDetector` | `PPDocLayoutDetector` | 医疗文档布局检测 |
| `result_formatter.py` | `MedicalResultFormatter` | `ResultFormatter` | 结果格式化 + RapidOCR 坐标提取 |
| `pipeline.py` | `MedicalOcrPipeline` | - | 集成三个组件的 Pipeline |

### 2. Utils（工具模块）

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `cache.py` | `SimpleCache` | 内存缓存（LRU）|
| `cache.py` | `RedisCache` | Redis 分布式缓存 |
| `config.py` | `load_medical_config()` | 配置加载 |
| `logger.py` | `setup_logging()` | 日志配置 |

### 3. High-Perf（高性能模块）

| 文件 | 类 | 功能 |
|------|------|------|
| `pipeline_pool.py` | `SingleGPUPipelinePool` | 单 GPU 多 Pipeline 池 |
| `pipeline_pool.py` | `MultiGPUPipelinePool` | 多 GPU Pipeline 池 |
| `high_perf_server.py` | FastAPI App | 高性能 REST API |

## 🚀 使用方式

### 基础用法

```python
from medical_ocr import (
    MedicalPageLoader,
    MedicalLayoutDetector,
    MedicalResultFormatter,
    MedicalOcrPipeline,
)
from glmocr.config import load_config

# 创建组件
page_loader = MedicalPageLoader(config, enable_preprocessing=True)
layout_detector = MedicalLayoutDetector(config)
result_formatter = MedicalResultFormatter(config)

# 创建 Pipeline
pipeline = MedicalOcrPipeline(
    config=config,
    page_loader=page_loader,
    layout_detector=layout_detector,
    result_formatter=result_formatter,
)

# 处理图像
pipeline.start()
results = list(pipeline.process(image_data))
pipeline.stop()
```

### 高性能用法

```python
from medical_ocr import create_pipeline_pool

# 创建 Pipeline 池
pool = create_pipeline_pool(
    pipeline_config=config.pipeline,
    pool_size=4,
    mode="single_gpu",  # 或 "multi_gpu"
    gpu_device_id=0,
)

# 初始化
pool.initialize()

# 并行处理
results = pool.process_batch(requests)

# 关闭
pool.shutdown()
```

### 高性能服务器

```python
from medical_ocr.high_perf_server import create_app

app = create_app()

# 启动
# uvicorn medical_ocr.high_perf_server:app --host 0.0.0.0 --port 8080
```

## 📊 配置

### config.yaml（完整配置）

包含所有可配置选项，适合生产环境。

### config_simple.yaml（简化配置）

快速上手配置，适合开发和测试。

## 📦 依赖

- `glmocr` - 基础 OCR 框架
- `fastapi` - 高性能 Web 框架
- `uvicorn` - ASGI 服务器
- `torch` - 深度学习框架
- `ultralytics` - YOLO 模型
- `rapidocr_onnxruntime` - RapidOCR 文字识别

## 🔧 开发

```bash
# 安装依赖
pip install -r requirements.txt

# 运行示例
python examples/basic_usage.py
python examples/high_perf_usage.py

# 运行测试
python run_tests.py
```

## 📝 导出 API

```python
from medical_ocr import __all__

print(__all__)
# ['MedicalPageLoader', 'MedicalLayoutDetector', 'MedicalResultFormatter',
#  'MedicalOcrPipeline', 'PipelinePool', 'SingleGPUPipelinePool',
#  'MultiGPUPipelinePool', 'PipelinePoolConfig', 'create_pipeline_pool',
#  'create_app', 'run_server', 'utils', 'constants']
```

## 🎓 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Medical OCR 架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              High-Perf Server (FastAPI)            │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │              Pipeline Pool Manager                  │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │   │
│  │  │  Pipeline 1 │ │  Pipeline 2 │ │  Pipeline N │  │   │
│  │  │  (GPU 0)    │ │  (GPU 0)    │ │  (GPU N)    │  │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘  │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │                 Medical Pipeline                      │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │  │PageLoader  │→│LayoutDetect │→│ResultFormat │    │   │
│  │  │(预处理)     │ │(布局检测)    │ │(结果格式化)  │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
