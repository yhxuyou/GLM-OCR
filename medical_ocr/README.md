# Medical OCR

专业的医疗文档 OCR 预处理和处理库，基于 glmocr 构建。

## 核心功能

### MedicalPageLoader - 核心预处理引擎

`MedicalPageLoader` 是本项目的核心类，继承自 glmocr 的 `PageLoader`，增加了三阶段文档预处理：

#### 1. YOLO 文档检测
- 使用 YOLO 模型检测文档区域
- 自动裁剪文档边界
- 支持自定义检测阈值

#### 2. RapidOCR 方向检测
- 检测文档倾斜角度
- 自动旋转矫正
- 支持 90°/180°/270° 旋转

#### 3. UVDoc 扭曲矫正
- 矫正文档透视畸变
- 恢复文档平整度
- 提供基础的透视矫正备选方案

## 快速开始

### 安装依赖

```bash
pip install glmocr opencv-python numpy Pillow
pip install ultralytics   # YOLO 模型支持
pip install rapidocr_onnxruntime  # RapidOCR 支持
```

### 基本使用

```python
from medical_ocr import MedicalPageLoader
from glmocr.config import PageLoaderConfig

# 创建配置
config = PageLoaderConfig()

# 创建 MedicalPageLoader
page_loader = MedicalPageLoader(
    config=config,
    yolo_model_dir="/path/to/yolo/model",
    uvdoc_model_dir="/path/to/uvdoc/model",
    enable_preprocessing=True
)

# 初始化模型
page_loader.start()

# 加载并预处理图片
pages = page_loader.load_pages(["medical_document.jpg"])

# 预处理单张图片
from PIL import Image
image = Image.open("medical_doc.jpg")
processed = page_loader.preprocess_image(image)

# 停止
page_loader.stop()
```

### 预处理流程

```
输入图片
  ↓
1. detect_document()      [YOLO 检测]
  ↓
2. crop_document()         [裁剪文档]
  ↓
3. correct_orientation()   [RapidOCR 方向矫正]
  ↓
4. correct_distortion()    [UVDoc 扭曲矫正]
  ↓
输出预处理后的图片
```

## API 参考

### MedicalPageLoader

#### 初始化参数

| 参数 | 类型 | 说明 |
|------|------|------|
| config | PageLoaderConfig | glmocr 配置对象 |
| yolo_model_dir | str | YOLO 模型目录路径 |
| uvdoc_model_dir | str | UVDoc 模型目录路径 |
| enable_preprocessing | bool | 是否启用预处理（默认 True） |

#### 主要方法

| 方法 | 说明 |
|------|------|
| `start()` | 初始化所有预处理模型 |
| `stop()` | 释放模型资源 |
| `load_pages(sources)` | 加载并预处理页面 |
| `preprocess_image(image)` | 预处理单张图片 |
| `detect_document(image)` | YOLO 文档检测 |
| `correct_orientation(image)` | RapidOCR 方向矫正 |
| `correct_distortion(image)` | UVDoc 扭曲矫正 |

#### 配置选项

```python
# YOLO 检测配置
page_loader.yolo_confidence_threshold = 0.5  # 置信度阈值
page_loader.yolo_nms_threshold = 0.45      # NMS 阈值
page_loader.document_padding = 10            # 裁剪边距（像素）

# 预处理控制
page_loader.enable_preprocessing = True      # 启用/禁用预处理
```

## 项目结构

```
medical_ocr/
├── medical_ocr/
│   ├── __init__.py              # 包入口
│   ├── page_loader.py           # MedicalPageLoader 核心类
│   ├── pipeline.py              # 处理流程编排
│   ├── layout_detector.py       # 布局检测器
│   ├── uvdoc_inference.py       # UVDoc 推理参考实现
│   └── server.py                # Flask 服务器
├── examples/
│   └── basic_usage.py           # 使用示例
├── tests/
├── pyproject.toml
└── README.md
```

## 模型要求

### YOLO 模型
- 模型文件：`best.pt`
- 支持的任务：文档检测（单类或多类）
- 推荐框架：ultralytics YOLOv8/v5

### UVDoc 模型
- 支持格式：.pt, .onnx
- 功能：文档透视矫正
- 参考实现：`uvdoc_inference.py`

## 许可

Apache-2.0
