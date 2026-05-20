# Medical OCR

专业的医疗文档 OCR 处理库，基于 glmocr 构建，保持原有的 Pipeline 推理逻辑不变，只替换了关键组件。

## 核心特点

✅ **不改变原有逻辑** - MedicalOcrPipeline 继承自 glmocr.Pipeline，所有 process() 方法完全继承  
✅ **替换关键组件** - 使用 MedicalPageLoader 和 MedicalLayoutDetector  
✅ **三阶段预处理** - YOLO 检测 -> 方向矫正 -> 畸变矫正  
✅ **可无缝替换** - 现有代码只需要替换 Pipeline 类即可

## 快速开始

### 安装依赖

```bash
pip install -e .
```

可选依赖（用于预处理）:

```bash
pip install ultralytics  # YOLO 文档检测
pip install rapidocr_onnxruntime  # 方向检测
pip install opencv-python
```

### 基本使用

与原始 glmocr.Pipeline 用法完全一致：

```python
from medical_ocr import MedicalOcrPipeline
from glmocr.config import load_config

# 加载配置
config = load_config()

# 创建医疗专用 Pipeline（唯一的不同之处！）
pipeline = MedicalOcrPipeline(
    config=config.pipeline,
    yolo_model_dir="/path/to/yolo/model",  # 可选
    uvdoc_model_dir="/path/to/uvdoc/model",  # 可选
)

# 以下代码与原始 Pipeline 完全相同！
pipeline.start()

for result in pipeline.process(request_data):
    print(result.json_result)
    print(result.markdown_result)

pipeline.stop()
```

### 预处理流程

MedicalPageLoader 自动在加载图片时执行三阶段预处理：

```
输入图片
    ↓
1. YOLO 文档检测 → 定位并裁剪文档区域
    ↓
2. RapidOCR 方向检测 → 旋转至正确角度
    ↓
3. UVDoc 畸变矫正 → 平坦化透视变形
    ↓
预处理后图片 → 交给 LayoutDetector 和 OCR
```

## 核心类说明

### MedicalPageLoader
继承自 `glmocr.dataloader.PageLoader`，增加预处理功能：

- `detect_document(image)` - YOLO 文档检测
- `correct_orientation(image)` - RapidOCR 方向矫正
- `correct_distortion(image)` - UVDoc 畸变矫正
- `preprocess_image(image)` - 完整预处理流程

### MedicalLayoutDetector
继承自 `glmocr.layout.PPDocLayoutDetector`，增加医疗特定优化：

- 医疗图像增强
- 医疗区域检测优化
- 多边形平滑

### MedicalOcrPipeline
继承自 `glmocr.pipeline.Pipeline`：

- **__init__ 唯一修改** - 替换 PageLoader 和 LayoutDetector
- **其他方法** - 完全继承自父类，无需修改

## 项目结构

```
medical_ocr/
├── medical_ocr/
│   ├── __init__.py          # 包入口
│   ├── page_loader.py       # MedicalPageLoader (⭐ 核心)
│   ├── layout_detector.py   # MedicalLayoutDetector
│   ├── pipeline.py          # MedicalOcrPipeline (继承自 glmocr.Pipeline)
│   └── uvdoc_inference.py   # UVDoc 推理包装
├── examples/
│   └── basic_usage.py       # 使用示例
├── tests/
├── pyproject.toml
└── README.md
```

## 迁移指南

从 glmocr 迁移到 medical-ocr，只需要：

### 原代码 (glmocr)
```python
from glmocr.pipeline import Pipeline

config = load_config()
pipeline = Pipeline(config.pipeline)
pipeline.start()
# ... 使用 ...
pipeline.stop()
```

### 新代码 (medical-ocr)
```python
from medical_ocr import MedicalOcrPipeline  # 只需要改这个

config = load_config()
pipeline = MedicalOcrPipeline(  # 只需要改这个
    config=config.pipeline,
    yolo_model_dir="/path/to/yolo",  # 可选
    uvdoc_model_dir="/path/to/uvdoc",  # 可选
)
pipeline.start()
# ... 使用完全相同！ ...
pipeline.stop()
```

## 许可证

Apache-2.0
