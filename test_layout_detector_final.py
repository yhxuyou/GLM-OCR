#!/usr/bin/env python3
"""测试layout_detector.py的实际执行结果"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'glmocr'))

def test_layout_detector():
    """测试实际的layout_detector"""
    try:
        from glmocr.layout.layout_detector import LayoutDetector
        
        print("加载layout_detector...")
        detector = LayoutDetector(model_dir=None)  # 使用模拟模式
        
        # 创建测试图片
        print("\n创建测试图片...")
        img = Image.new('RGB', (1200, 800), color='white')
        
        # 模拟检测结果
        mock_detections = [
            {
                "label": "table",
                "box": [100, 100, 600, 300],  # 相对较宽
                "score": 0.95,
                "polygon_points": [[100, 100], [600, 100], [600, 300], [100, 300]],
            },
            {
                "label": "table",
                "box": [200, 350, 500, 750],  # 相对较高
                "score": 0.90,
                "polygon_points": [[200, 350], [500, 350], [500, 750], [200, 750]],
            },
        ]
        
        print(f"模拟检测到 {len(mock_detections)} 个表格")
        for i, det in enumerate(mock_detections):
            print(f"  表格 {i+1}: {det['box']}")
        
        # 调用处理函数
        print("\n处理图片...")
        results, vis_images = detector.process_image(img, mock_detections)
        
        print(f"\n处理结果:")
        print(f"  总区域数: {len(results)}")
        
        table_count = sum(1 for r in results if r['label'] == 'table')
        text_count = sum(1 for r in results if r['label'] == 'text')
        
        print(f"  表格区域: {table_count}")
        print(f"  文本区域: {text_count}")
        
        for i, result in enumerate(results):
            print(f"\n  区域 {i+1}:")
            print(f"    标签: {result['label']}")
            print(f"    bbox: {result['bbox_2d']}")
            print(f"    task_type: {result['task_type']}")
        
        return len(results) > 0
        
    except ImportError as e:
        print(f"无法导入layout_detector: {e}")
        print("\n创建独立的测试...")
        test_standalone()
        return False

def test_standalone():
    """独立测试逻辑"""
    from PIL import Image, ImageDraw, ImageFont
    
    print("="*70)
    print("独立测试layout_detector.py的表格扩展逻辑")
    print("="*70)
    
    # 模拟检测到的表格
    table_regions = [
        {
            "box": [100, 100, 600, 300],  # 相对较宽的表格
            "label": "table",
            "score": 0.95,
            "polygon_points": [[100, 100], [600, 100], [600, 300], [100, 300]],
        },
        {
            "box": [200, 350, 500, 750],  # 相对较高的表格
            "label": "table",
            "score": 0.90,
            "polygon_points": [[200, 350], [500, 350], [500, 750], [200, 750]],
        },
    ]
    
    image_width, image_height = 1200, 800
    
    print(f"\n图片尺寸: {image_width}x{image_height}")
    print(f"检测到的表格: {len(table_regions)}")
    
    # 执行表格扩展逻辑（复制layout_detector.py的代码）
    expand_margin = 20
    expand_margin_x = max(expand_margin, int(image_width * 0.02))
    expand_margin_y = max(expand_margin, int(image_height * 0.02))
    
    # Step 3: 确定扩展方向
    extended_tables = []
    for table in table_regions:
        x1, y1, x2, y2 = table["box"]
        table_width = x2 - x1
        table_height = y2 - y1
        
        width_ratio = table_width / image_width
        height_ratio = table_height / image_height
        
        # 修复后的条件：使用 >=
        if width_ratio >= height_ratio:
            ext_x1, ext_y1, ext_x2, ext_y2 = 0, y1, image_width, y2
            direction = "左右扩展"
        else:
            ext_x1, ext_y1, ext_x2, ext_y2 = x1, 0, x2, image_height
            direction = "上下扩展"
        
        extended_tables.append({
            "original_box": table["box"],
            "extended_box": [ext_x1, ext_y1, ext_x2, ext_y2],
            "label": table["label"],
            "score": table["score"],
            "polygon_points": table["polygon_points"],
        })
        
        print(f"\n表格: ({x1}, {y1}, {x2}, {y2})")
        print(f"  尺寸: {table_width}x{table_height}")
        print(f"  宽占比: {width_ratio:.2%}, 高占比: {height_ratio:.2%}")
        print(f"  → {direction}")
        print(f"  扩展: ({ext_x1}, {ext_y1}, {ext_x2}, {ext_y2})")
    
    # Step 4: 确定分割模式
    is_landscape = image_width > image_height
    wide_table_count = sum(1 for table in table_regions if (table["box"][2] - table["box"][0]) / image_width > (table["box"][3] - table["box"][1]) / image_height)
    use_horizontal_split = (wide_table_count >= len(table_regions) / 2) if table_regions else is_landscape
    
    print(f"\n分割模式: {'水平分割' if use_horizontal_split else '垂直分割'}")
    
    # Step 5: 排序
    if use_horizontal_split:
        extended_tables.sort(key=lambda t: t["extended_box"][0])
    else:
        extended_tables.sort(key=lambda t: t["extended_box"][1])
    
    print(f"排序后的表格:")
    for i, table in enumerate(extended_tables):
        print(f"  {i+1}: {table['extended_box']}")
    
    # Step 6: 生成文本区域
    text_regions = []
    current_pos = 0
    
    if use_horizontal_split:
        for table in extended_tables:
            ext_x1, ext_y1, ext_x2, ext_y2 = table["extended_box"]
            if current_pos < ext_x1:
                text_regions.append([current_pos, 0, ext_x1, image_height])
                print(f"\n添加文本区域: ({current_pos}, 0, {ext_x1}, {image_height})")
            current_pos = ext_x2
        if current_pos < image_width:
            text_regions.append([current_pos, 0, image_width, image_height])
            print(f"\n添加文本区域: ({current_pos}, 0, {image_width}, {image_height})")
    else:
        for table in extended_tables:
            ext_x1, ext_y1, ext_x2, ext_y2 = table["extended_box"]
            if current_pos < ext_y1:
                text_regions.append([0, current_pos, image_width, ext_y1])
                print(f"\n添加文本区域: (0, {current_pos}, {image_width}, {ext_y1})")
            current_pos = ext_y2
        if current_pos < image_height:
            text_regions.append([0, current_pos, image_width, image_height])
            print(f"\n添加文本区域: (0, {current_pos}, {image_width}, {image_height})")
    
    print(f"\n" + "="*70)
    print(f"结果总结:")
    print(f"  表格区域: {len(table_regions)}")
    print(f"  文本区域: {len(text_regions)}")
    
    if len(text_regions) == 0:
        print("\n⚠️  警告: 没有生成文本区域！")
        print("\n原因分析:")
        print("  表格1: 左右扩展，填满整个宽度")
        print("  表格2: 上下扩展，填满整个高度")
        print("  → 两个表格覆盖了大部分区域，没有剩余空间作为文本区域")
        
        print("\n解决方案:")
        print("  方案1: 确保有至少一个方向的表格不填满整个页面")
        print("  方案2: 调整表格的宽高比")
        print("  方案3: 如果确实需要文本区域，考虑添加额外的文本区域覆盖整个页面")
    else:
        print("\n✅ 成功生成文本区域")
        for i, region in enumerate(text_regions):
            print(f"  文本区域 {i+1}: {region}")

if __name__ == "__main__":
    test_layout_detector()
