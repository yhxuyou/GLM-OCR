# 医疗费用清单处理器使用指南

## 概述

医疗费用清单处理器是 GLM-OCR 的定制化扩展，专门针对医疗费用清单的场景进行优化，实现：

1. **表格区域识别和扩展裁剪** - 智能识别医疗费用明细表，并向外扩展5%（最小10像素）保留边框信息
2. **非表格区域整体合并** - 将标题、患者信息、诊断说明等非表格内容合并为整体图像
3. **结构化输出** - 提供标准化的数据结构，便于后续处理

## 目录结构

```
deploy/
├── medical_expense_processor.py      # 核心处理器
├── medical_expense_pipeline.py       # Pipeline集成适配器
├── test_medical_expense.py          # 测试脚本
└── README_MEDICAL_EXPENSE.md        # 本文档
```

## 快速开始

### 1. 基本使用

```python
from PIL import Image
from deploy.medical_expense_processor import MedicalExpenseProcessor

# 加载图像
image = Image.open("medical_expense_list.jpg")

# 定义布局检测结果（通常由GLM-OCR的layout检测器提供）
layout_results = [
    {
        "label": "table",           # 表格标签
        "bbox_2d": [100, 200, 500, 600],  # 归一化坐标 (0-1000)
        "score": 0.95,              # 置信度
        "polygon": [[100, 200], [500, 200], [500, 600], [100, 600]]
    },
    {
        "label": "text",
        "bbox_2d": [50, 50, 950, 150],
        "score": 0.90
    }
]

# 创建处理器
processor = MedicalExpenseProcessor(
    table_expand_ratio=0.05,    # 表格扩展5%
    table_min_expansion=10      # 最小扩展10像素
)

# 处理图像
result = processor.process(image, layout_results)

# 访问结果
print(f"检测到 {len(result.table_regions)} 个表格")
print(f"表格坐标: {result.table_regions[0].bbox}")
print(f"扩展后坐标: {result.table_regions[0].expanded_bbox}")

# 保存结果
saved_files = result.save("./output")
print(f"已保存到: {saved_files}")
```

### 2. 与Pipeline集成

```python
from deploy.medical_expense_pipeline import MedicalExpensePipelineAdapter

# 创建集成Pipeline
adapter = MedicalExpensePipelineAdapter(
    config_path="glmocr/config.yaml",
    table_labels=["table", "表格", "fee_table"],  # 自定义表格标签
    table_expand_ratio=0.05,
    table_min_expansion=10
)

# 启动
adapter.start()

# 处理图像
result = adapter.parse(
    "medical_expense_list.jpg",
    return_medical_expense_result=True
)

# 获取标准OCR结果
print(result["markdown_result"])

# 获取医疗费用清单专用结果
print(result["medical_expense"])

# 获取表格裁剪图
medical_result = adapter.get_medical_result("medical_expense_list.jpg")
for table in medical_result.table_regions:
    if table.cropped_image:
        table.cropped_image.save(f"table_{table.table_id}.png")

# 停止
adapter.stop()
```

## 核心功能

### 1. 表格区域扩展裁剪

**问题**: 医疗费用清单中的表格通常有边框、表头装饰等，这些信息位于表格检测框的外侧，如果严格按检测框裁剪会丢失这些重要信息。

**解决方案**: 向外扩展5%的宽度和高度，最小扩展10像素。

```
原始表格框         扩展后（+5%）
┌─────────┐       ┌───────────┐
│  表格   │  -->  │  ┌─────┐ │
│         │       │  │表格 │ │
└─────────┘       │  └─────┘ │
                  └───────────┘
```

**代码示例**:

```python
processor = MedicalExpenseProcessor(
    table_expand_ratio=0.05,    # 扩展5%
    table_min_expansion=10      # 最小10像素
)
```

### 2. 非表格区域合并

**问题**: 医疗费用清单中除了表格，还有大量非表格内容（标题、患者信息、诊断说明、备注等），这些内容需要保持相对空间位置。

**解决方案**: 从原图中"挖掉"所有表格区域，将剩余部分合并为一张整体图像。

```
原图                      合并后（非表格区域）
┌────────────────┐        ┌────────────────┐
│     标题       │        │     标题       │
├────────────────┤   -->  ├────────────────┤
│                │        │                │
│    表格区域    │        │ (白色填充)     │
│                │        │                │
├────────────────┤        ├────────────────┤
│     落款       │        │     落款       │
└────────────────┘        └────────────────┘
```

### 3. 支持多表格识别

```python
layout_results = [
    {"label": "table", "bbox_2d": [100, 200, 500, 600]},
    {"label": "table", "bbox_2d": [100, 700, 500, 1000]},  # 第二个表格
    {"label": "table", "bbox_2d": [550, 200, 950, 600]},   # 第三个表格
    {"label": "text", "bbox_2d": [50, 50, 950, 150]},
]

result = processor.process(image, layout_results)

print(f"共检测到 {len(result.table_regions)} 个表格")
for i, table in enumerate(result.table_regions):
    print(f"表格{i+1}: 坐标={table.bbox}, 扩展后={table.expanded_bbox}")
```

## 配置参数

### MedicalExpenseProcessor

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `table_expand_ratio` | float | 0.05 | 表格扩展比例 (0.05 = 5%) |
| `table_min_expansion` | int | 10 | 最小扩展像素数 |
| `save_intermediate` | bool | False | 是否保存中间结果 |

### MedicalExpensePipelineAdapter

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config_path` | str | None | GLM-OCR配置文件路径 |
| `table_labels` | List[str] | ["table", "表格"] | 表格标签列表 |
| `table_expand_ratio` | float | 0.05 | 表格扩展比例 |
| `table_min_expansion` | int | 10 | 最小扩展像素 |

## 数据结构

### TableRegion

```python
@dataclass
class TableRegion:
    table_id: int                      # 表格ID
    bbox: Tuple[int,int,int,int]      # 原始边界框 (像素坐标)
    bbox_norm: Tuple[int,int,int,int] # 归一化坐标 (0-1000)
    label: str                         # 标签
    score: float                       # 置信度
    polygon: Optional[List]            # 多边形点
    expanded_bbox: Optional[Tuple]     # 扩展后的边界框
    cropped_image: Optional[Image]     # 裁剪后的图像
```

### MedicalExpenseResult

```python
@dataclass
class MedicalExpenseResult:
    table_regions: List[TableRegion]   # 表格区域列表
    non_table_image: NonTableImage     # 非表格整体图像
    original_image: Image              # 原始图像
    image_size: Tuple[int,int]         # 图像尺寸

    def save(output_dir: str) -> Dict[str,str]:
        """保存所有结果"""
```

## 输出示例

```python
# 处理后的结果
{
    "table_count": 2,
    "table_regions": [
        {
            "table_id": 0,
            "bbox": (200, 400, 800, 1200),
            "bbox_norm": [100, 200, 400, 600],
            "label": "table",
            "score": 0.95,
            "expanded_bbox": (180, 380, 820, 1220),
            "has_cropped_image": true
        },
        {
            "table_id": 1,
            "bbox": (200, 1300, 800, 2000),
            "bbox_norm": [100, 650, 400, 1000],
            "label": "table",
            "score": 0.92,
            "expanded_bbox": (180, 1280, 820, 2020),
            "has_cropped_image": true
        }
    ],
    "has_non_table_image": true,
    "non_table_image_info": {
        "original_size": (2000, 2500),
        "cropped_regions_count": 3,
        "image_size": (2000, 2500)
    },
    "image_size": (2000, 2500)
}
```

## 运行测试

```bash
# 运行所有测试
python deploy/test_medical_expense.py

# 查看测试输出
python deploy/test_medical_expense.py 2>&1 | tee test_output.log
```

## 与FastAPI集成

可以在 FastAPI 服务中添加医疗费用清单处理的端点：

```python
from fastapi import UploadFile, File
from deploy.medical_expense_processor import MedicalExpenseProcessor

@app.post("/api/v1/medical-expense/parse")
async def parse_medical_expense(
    file: UploadFile = File(...),
    layout_results: str = Form(...)  # JSON字符串
):
    # 读取图像
    image = Image.open(BytesIO(await file.read()))
    
    # 解析布局结果
    layouts = json.loads(layout_results)
    
    # 处理
    processor = MedicalExpenseProcessor()
    result = processor.process(image, layouts)
    
    # 返回结果
    return result.to_dict()
```

## 常见问题

### Q: 如何调整表格扩展量？

A: 根据实际效果调整参数：

```python
# 扩展更多（保留更多边框信息）
processor = MedicalExpenseProcessor(
    table_expand_ratio=0.08,    # 8%
    table_min_expansion=15      # 最小15像素
)

# 扩展更少
processor = MedicalExpenseProcessor(
    table_expand_ratio=0.03,    # 3%
    table_min_expansion=5       # 最小5像素
)
```

### Q: 如何识别不同类型的表格？

A: 通过 `table_labels` 参数指定：

```python
processor = MedicalExpenseProcessor()

layout_results = [
    {"label": "fee_detail_table", "bbox_2d": [...]},      # 费用明细表
    {"label": "summary_table", "bbox_2d": [...]},          # 汇总表
    {"label": "diagnosis_table", "bbox_2d": [...]},       # 诊断信息表
]

result = processor.process(
    image,
    layout_results,
    table_labels=["fee_detail_table", "summary_table", "diagnosis_table"]
)
```

### Q: 非表格区域合并后表格区域显示为白色吗？

A: 是的，表格区域会被填充为白色 (RGB: 255, 255, 255)。如果需要其他填充颜色，可以修改 `_merge_non_table_regions` 方法。

## 性能优化建议

1. **批量处理**: 使用 Pipeline 的并发特性

2. **内存管理**: 处理完成后及时释放结果

```python
result = processor.process(image, layouts)

# 保存需要的部分
for table in result.table_regions:
    table.cropped_image.save(f"table_{table.table_id}.png")

# 清理大对象
result.non_table_image.merged_image = None
```

3. **多进程**: 对于大量图像，使用 multiprocessing

## 许可证

与 GLM-OCR 项目使用相同的许可证。
