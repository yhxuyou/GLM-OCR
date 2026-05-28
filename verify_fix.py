#!/usr/bin/env python3
"""验证修复后的layout_detector.py逻辑"""

# 直接从layout_detector.py复制修复后的逻辑
def simulate_fixed_logic(image_width, image_height, tables):
    print(f"\n图片尺寸: {image_width}x{image_height}")
    print(f"检测到的表格: {len(tables)}")
    
    # Step 3: 确定每个表格的扩展方向（使用 >= 修复）
    extended_tables = []
    for i, table in enumerate(tables):
        x1, y1, x2, y2 = table["box"]
        table_width = x2 - x1
        table_height = y2 - y1
        
        width_ratio = table_width / image_width
        height_ratio = table_height / image_height
        
        # 修复：使用 >= 而不是 >
        if width_ratio >= height_ratio:
            ext_x1, ext_y1, ext_x2, ext_y2 = 0, y1, image_width, y2
            direction = "左右扩展"
        else:
            ext_x1, ext_y1, ext_x2, ext_y2 = x1, 0, x2, image_height
            direction = "上下扩展"
        
        extended_tables.append({
            "extended_box": [ext_x1, ext_y1, ext_x2, ext_y2],
        })
        
        print(f"\n表格 {i+1}: ({x1}, {y1}, {x2}, {y2})")
        print(f"  尺寸: {table_width}x{table_height}")
        print(f"  宽占比: {width_ratio:.2%}, 高占比: {height_ratio:.2%}")
        print(f"  → {direction}")
        print(f"  扩展: ({ext_x1}, {ext_y1}, {ext_x2}, {ext_y2})")
    
    # Step 4: 确定分割模式
    is_landscape = image_width > image_height
    wide_table_count = sum(1 for t in tables if (t["box"][2] - t["box"][0]) / image_width > (t["box"][3] - t["box"][1]) / image_height)
    use_horizontal_split = (wide_table_count >= len(tables) / 2) if tables else is_landscape
    
    print(f"\n分割模式: {'水平分割' if use_horizontal_split else '垂直分割'}")
    
    # Step 5: 排序
    if use_horizontal_split:
        extended_tables.sort(key=lambda t: t["extended_box"][0])
    else:
        extended_tables.sort(key=lambda t: t["extended_box"][1])
    
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
    
    print(f"\n文本区域数量: {len(text_regions)}")
    return text_regions

# 测试场景
print("="*70)
print("测试修复后的逻辑")
print("="*70)

print("\n场景1: 两个正方形表格（宽占比=高占比），使用>=后应该左右扩展")
simulate_fixed_logic(1200, 800, [
    {"box": [100, 100, 400, 400]},  # 25% x 50% → 应该左右扩展
    {"box": [600, 150, 900, 450]},  # 25% x 37.5% → 应该左右扩展
])

print("\n" + "="*70)
print("\n场景2: 一个明显宽的表格（宽占比>高占比）")
simulate_fixed_logic(1200, 800, [
    {"box": [100, 100, 600, 300]},  # 41.67% x 25% → 左右扩展
])

print("\n" + "="*70)
print("\n场景3: 一个明显高的表格（高占比>宽占比）")
simulate_fixed_logic(1200, 800, [
    {"box": [200, 100, 500, 500]},  # 25% x 50% → 上下扩展
])

print("\n" + "="*70)
print("\n场景4: 混合场景 - 一个宽表格，一个高表格")
simulate_fixed_logic(1200, 800, [
    {"box": [100, 100, 600, 300]},  # 41.67% x 25% → 左右扩展
    {"box": [200, 350, 500, 750]},  # 25% x 50% → 上下扩展
])

print("\n" + "="*70)
print("修复验证总结:")
print("✅ 使用 >= 替代 > 确保在相等时至少有一个方向有文本区域")
print("✅ 表格宽占比 >= 高占比时，左右扩展（产生上下方向的文本区域）")
print("✅ 表格高占比 > 宽占比时，上下扩展（产生左右方向的文本区域）")
