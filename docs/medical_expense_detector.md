# 医疗费用清单布局检测器 (Medical Expense Layout Detector)

## 概述

`MedicalExpenseLayoutDetector` 是基于 PP-DocLayoutV3 优化的医疗费用清单专属布局检测器，支持：

1. **表格区域识别和扩展裁剪** - 识别表格区域并向外扩展5%（最小10像素），保留表格边框信息
2. **非表格区域整体合并** - 将所有非表格内容（标题、患者信息、诊断说明等）合并为一张整体图像
3. **结构化输出** - 输出表格裁剪图列表 + 非表格整体图

## 文件位置

- 核心文件: [`glmocr/layout/medical_detector.py`](file:///workspace/glmocr/layout/medical_detector.py)
- 示例文件: [`examples/medical_expense_detector_example.py`](file:///workspace/examples/medical_expense_detector_example.py)

## 快速开始

### 1. 基本使用

```python
from PIL import Image
from glmocr.config import load_config
from glmocr.layout.medical_detector import create_medical_expense_detector

# 1. 加载配置
config = load_config()

# 2. 创建医疗费用清单检测器
detector = create_medical_expense_detector(
    config.layout,
    table_expand_ratio=0.05,    # 表格扩展5%
    table_min_expansion=10,     # 最小扩展10像素
    table_labels=["table", "表格"],  # 表格标签
)

# 3. 启动检测器
detector.start()

try:
    # 4. 加载图像
    image = Image.open("medical_expense.jpg")

    # 5. 处理图像
    results, vis, medical_results = detector.process(
        [image],
        save_visualization=True,
        return_medical_result=True,
    )

    # 6. 获取医疗费用清单结果
    medical_result = medical_results[0]

    # 7. 处理表格
    for table in medical_result.table_regions:
        print(f"表格 {table.table_id}:")
        print(f"  原始坐标: {table.bbox_pixel}")
        print(f"  扩展坐标: {table.expanded_bbox_pixel}")
        if table.cropped_image:
            table.cropped_image.save(f"table_{table.table_id}.png")

    # 8. 处理非表格整体图
    if medical_result.non_table_image:
        medical_result.non_table_image.merged_image.save("non_table.png")

    # 9. 一键保存所有结果
    medical_result.save("./output")

finally:
    # 10. 停止检测器
    detector.stop()
```

### 2. 使用命令行示例

```bash
# 使用默认配置
python examples/medical_expense_detector_example.py --image medical_expense.jpg --output ./output

# 自定义扩展参数
python examples/medical_expense_detector_example.py \
    --image medical_expense.jpg \
    --output ./output \
    --expand-ratio 0.08 \
    --min-expansion 15
```

## 核心功能

### 1. 表格区域识别和扩展裁剪

**原理**: 检测到表格区域后，向四周扩展一定比例（默认5%），最小扩展10像素，确保边框信息保留。

**代码示例**:
```python
# 设置扩展参数
detector = create_medical_expense_detector(
    config.layout,
    table_expand_ratio=0.05,  # 5%
    table_min_expansion=10,   # 10像素
)

# 获取扩展后的表格图像
for table in medical_result.table_regions:
    if table.cropped_image:
        table.cropped_image.save(f"table_{table.table_id}.png")
```

### 2. 非表格区域整体合并

**原理**: 从原图中"挖掉"所有表格区域（填充为白色），保留非表格内容的相对空间位置。

**代码示例**:
```python
# 获取非表格整体图
if medical_result.non_table_image:
    merged_img = medical_result.non_table_image.merged_image
    merged_img.save("non_table.png")

# 自定义填充颜色
detector = create_medical_expense_detector(
    config.layout,
    fill_color=(255, 255, 255),  # 白色填充
)
```

### 3. 完整结果保存

```python
# 一键保存所有结果
saved_files = medical_result.save(
    output_dir="./output",
    format="PNG"  # 支持 PNG, JPEG 等格式
)

# 查看保存的文件
print(saved_files)
# 输出:
# {
#   "table_000": "./output/table_000.png",
#   "table_001": "./output/table_001.png",
#   "non_table_regions": "./output/non_table_regions.png",
#   "original": "./output/original.png"
# }
```

## API 参考

### MedicalExpenseLayoutDetector

```python
MedicalExpenseLayoutDetector(config: LayoutConfig)
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `config` | `LayoutConfig` | - | 布局检测器配置 |
| `table_expand_ratio` | `float` | `0.05` | 表格扩展比例 |
| `table_min_expansion` | `int` | `10` | 最小扩展像素 |
| `table_labels` | `List[str]` | `["table", "表格"]` | 表格标签列表 |
| `fill_color` | `Tuple[int, int, int]` | `(255, 255, 255)` | 非表格区域中表格部分的填充色 |

#### 方法

##### `start()`
启动检测器，加载模型和处理器。

##### `stop()`
停止检测器，卸载模型释放资源。

##### `process(images, save_visualization=False, global_start_idx=0, use_polygon=False, return_medical_result=True)`

处理图像列表，返回布局检测结果。

**参数**:
- `images`: List[Image.Image] - PIL 图像列表
- `save_visualization`: bool - 是否生成可视化图像
- `global_start_idx`: int - 可视化页面编号起始索引
- `use_polygon`: bool - 是否使用多边形掩码
- `return_medical_result`: bool - 是否返回医疗费用清单特殊结果

**返回**:
- `Tuple[List[List[Dict]], Dict[int, Image.Image], List[MedicalExpenseResult]]`
  - 标准布局检测结果
  - 可视化图像字典
  - MedicalExpenseResult 列表

### MedicalExpenseResult

医疗费用清单处理结果。

#### 属性

| 属性 | 类型 | 说明 |
|-----|------|------|
| `table_regions` | `List[TableRegion]` | 表格区域列表 |
| `non_table_image` | `Optional[NonTableImage]` | 非表格整体图像 |
| `original_image` | `Optional[Image.Image]` | 原始图像 |
| `image_size` | `Tuple[int, int]` | 图像尺寸 |
| `all_regions` | `Optional[List[Dict]]` | 所有原始布局检测结果 |

#### 方法

##### `to_dict() -> Dict`
转换为字典格式。

##### `save(output_dir: str, format: str = "PNG") -> Dict[str, str]`
保存所有结果到指定目录。

### TableRegion

表格区域数据结构。

| 属性 | 类型 | 说明 |
|-----|------|------|
| `table_id` | `int` | 表格ID |
| `bbox_2d` | `List[int]` | 归一化坐标 (0-1000) |
| `bbox_pixel` | `Tuple[int, int, int, int]` | 像素坐标 |
| `expanded_bbox_2d` | `Optional[List[int]]` | 扩展后归一化坐标 |
| `expanded_bbox_pixel` | `Optional[Tuple[int, int, int, int]]` | 扩展后像素坐标 |
| `label` | `str` | 标签 |
| `score` | `float` | 置信度 |
| `polygon` | `Optional[List[List[float]]]` | 多边形点 |
| `cropped_image` | `Optional[Image.Image]` | 裁剪后的图像 |

### NonTableImage

非表格区域整体图像数据结构。

| 属性 | 类型 | 说明 |
|-----|------|------|
| `merged_image` | `Image.Image` | 合并后的图像 |
| `original_size` | `Tuple[int, int]` | 原始尺寸 |
| `cropped_regions_count` | `int` | 裁剪的表格区域数量 |

## 完整工作流示例

```python
from PIL import Image
from glmocr.config import load_config
from glmocr.layout.medical_detector import create_medical_expense_detector

# 1. 加载配置
config = load_config()

# 2. 创建检测器（自定义参数）
detector = create_medical_expense_detector(
    config.layout,
    table_expand_ratio=0.05,    # 扩展5%
    table_min_expansion=10,     # 最小10像素
    table_labels=["table", "表格", "fee_table"],  # 多个表格标签
    fill_color=(255, 255, 255),  # 白色填充
)

# 3. 启动检测器
detector.start()

try:
    # 4. 加载多张图像（批量处理）
    images = [
        Image.open("medical1.jpg"),
        Image.open("medical2.jpg"),
    ]

    # 5. 批量处理
    results, vis, medical_results = detector.process(
        images,
        save_visualization=True,
        return_medical_result=True,
    )

    # 6. 处理每张图像的结果
    for idx, medical_result in enumerate(medical_results):
        print(f"\n图像 {idx}:")
        print(f"  检测到 {len(medical_result.table_regions)} 个表格")

        # 保存到单独目录
        output_dir = f"./output/image_{idx}"
        saved_files = medical_result.save(output_dir)

        print(f"  已保存到: {output_dir}")
        for name, path in saved_files.items():
            print(f"    {name}: {path}")

finally:
    # 7. 停止检测器
    detector.stop()
```

## 常见问题

### Q: 如何调整表格扩展比例？
A: 通过 `table_expand_ratio` 参数调整：
```python
detector = create_medical_expense_detector(
    config.layout,
    table_expand_ratio=0.1,  # 扩展10%
    table_min_expansion=20, # 最小20像素
)
```

### Q: 如何识别多种表格类型？
A: 通过 `table_labels` 参数指定：
```python
detector = create_medical_expense_detector(
    config.layout,
    table_labels=[
        "table", "表格",
        "fee_table", "费用表",
        "summary_table", "汇总表"
    ]
)
```

### Q: 如何修改非表格区域中表格部分的填充色？
A: 通过 `fill_color` 参数：
```python
detector = create_medical_expense_detector(
    config.layout,
    fill_color=(200, 200, 200),  # 灰色填充
)
```

### Q: 只需要标准布局检测结果，不需要医疗费用清单特殊结果？
A: 设置 `return_medical_result=False`：
```python
results, vis = detector.process(
    [image],
    return_medical_result=False,
)
```

## 性能优化建议

1. **GPU 配置**: 确保使用 GPU 加速，设置 `config.layout.cuda_visible_devices`
2. **批处理**: 一次处理多张图像，提高效率
3. **及时释放**: 使用后及时调用 `stop()` 释放显存
4. **阈值调整**: 根据实际情况调整检测阈值，减少误检

## 示例输出目录结构

```
output/
├── original.png              # 原始图像
├── non_table_regions.png     # 非表格整体图像（表格区域已白色填充）
├── table_000.png             # 表格0（已扩展裁剪）
├── table_001.png             # 表格1
├── table_002.png             # 表格2
└── layout_visualization.png  # 布局可视化（可选）
```
