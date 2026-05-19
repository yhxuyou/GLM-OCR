# Medical OCR

专业的医疗文档 OCR 处理库，基于 glmocr 构建，增加了医疗文档特定的优化功能。

## 功能特性

### 医疗文档优化

- **医疗图像增强**: 针对医疗文档（病历、化验单、X光片等）的专用图像预处理
- **布局检测优化**: 优先检测医疗相关的区域（表格、签名、印章等）
- **医学术语标准化**: 自动标准化医学术语和单位表示
- **OCR 结果清理**: 专门解决医疗文档中常见的 OCR 错误

### 技术特点

- 完全继承自 glmocr，保持 API 兼容性
- 不修改 glmocr 源码，通过继承方式扩展
- 支持自定义预处理和后处理钩子
- Flask 服务器提供 REST API 接口

## 安装

```bash
# 基础安装
pip install -e .

# 安装服务器功能
pip install -e ".[server]"

# 开发安装
pip install -e ".[dev]"
```

## 快速使用

### 作为库使用

```python
from glmocr.config import load_config
from medical_ocr.pipeline import MedicalOcrPipeline
from medical_ocr.layout_detector import MedicalLayoutDetector

# 加载配置
config = load_config()

# 创建医疗专用布局检测器
layout_detector = MedicalLayoutDetector(config.pipeline.layout)

# 创建医疗 OCR pipeline
pipeline = MedicalOcrPipeline(
    config=config.pipeline,
    layout_detector=layout_detector
)

# 添加自定义预处理钩子
def my_custom_preprocess(image, context):
    # 自定义图像处理逻辑
    return image

pipeline.add_preprocess_hook(my_custom_preprocess)

# 处理文档
pipeline.start()
results = list(pipeline.process(request_data))
for result in results:
    print(result.json_result)
pipeline.stop()
```

### 启动服务器

```bash
# 使用命令行启动
medical-ocr-server --host 0.0.0.0 --port 8080

# 或者直接运行模块
python -m medical_ocr.server
```

### API 调用

#### 健康检查

```bash
curl http://localhost:8080/health
```

#### 基础 OCR

```bash
curl -X POST http://localhost:8080/medical-ocr/parse \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["path/to/medical/document.jpg"],
    "preprocess_options": {
      "enable_enhancement": true
    },
    "postprocess_options": {
      "normalize_terminology": true
    }
  }'
```

#### 增强 OCR

```bash
curl -X POST http://localhost:8080/medical-ocr/parse/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["path/to/medical/document.jpg"],
    "medical_enhancements": {
      "enable_enhancement": true,
      "normalize_units": true,
      "extract_patient_info": true
    }
  }'
```

## 项目结构

```
medical_ocr/
├── medical_ocr/
│   ├── __init__.py          # 包入口
│   ├── pipeline.py          # MedicalOcrPipeline
│   ├── layout_detector.py   # MedicalLayoutDetector
│   └── server.py            # Flask 服务器
├── examples/                # 示例代码
├── tests/                   # 测试代码
├── pyproject.toml           # 项目配置
└── README.md               # 说明文档
```

## 核心类

### MedicalLayoutDetector

继承自 `PPDocLayoutDetector`，增加了医疗文档特定的功能：

- `_enhance_medical_image()`: 医疗图像增强（去噪、对比度增强）
- `_filter_medical_regions()`: 医疗区域过滤和优先级排序
- `_smooth_polygons()`: 多边形平滑处理

### MedicalOcrPipeline

继承自 `Pipeline`，增加了：

- 预处理钩子系统
- 后处理钩子系统
- 医学术语标准化
- 医疗文本清理

## 自定义扩展

### 添加自定义预处理

```python
def my_preprocessing(image, context):
    # 图像处理逻辑
    return processed_image

pipeline.add_preprocess_hook(my_preprocessing)
```

### 添加自定义后处理

```python
def my_postprocessing(json_result, markdown_result, context):
    # 结果处理逻辑
    return enhanced_json, enhanced_markdown

pipeline.add_postprocess_hook(my_postprocessing)
```

## 配置选项

可以通过 config 对象设置以下选项：

```python
# MedicalLayoutDetector 配置
config.pipeline.layout.enable_medical_enhancement = True
config.pipeline.layout.min_medical_region_area = 300
config.pipeline.layout.max_overlap_ratio = 0.85
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest
```

## 许可证

Apache-2.0
