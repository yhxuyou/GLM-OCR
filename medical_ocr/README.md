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

## 许可证

Apache-2.0
