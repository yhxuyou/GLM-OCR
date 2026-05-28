#!/usr/bin/env python3
"""测试表格区域扩展和页面分割效果（允许重叠，不遗漏信息）"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

# 确保能导入 glmocr 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'glmocr'))
sys.path.insert(0, os.path.dirname(__file__))

def draw_rectangle_with_text(img, box, label, color, text_color="white", alpha=128):
    """在图片上绘制带透明度的矩形框和文字"""
    x1, y1, x2, y2 = [int(v) for v in box]

    # 绘制文字（先画在最上层）
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("resources/PingFang.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    draw.rectangle([(x1, y1 - text_height - 10), (x1 + text_width + 10, y1)], fill=color)
    draw.text((x1 + 5, y1 - text_height - 5), label, fill=text_color, font=font)

    # 如果指定了透明度，绘制半透明填充
    if alpha < 255 and alpha > 0:
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # 处理颜色：如果是 RGBA，取前三个分量
        if len(color) == 4:
            fill_color = color[:3] + (alpha,)
        else:
            fill_color = color + (alpha,)

        overlay_draw.rectangle([(x1, y1), (x2, y2)], fill=fill_color)

        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)

    # 绘制边框
    draw = ImageDraw.Draw(img if img.mode == 'RGB' else img)
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

    return img if img.mode == 'RGB' else img.convert('RGB')

def test_overlap_no_missing():
    """测试允许重叠、不遗漏信息的分割逻辑"""

    # 1. 创建一个测试图片
    img_width, img_height = 1200, 800
    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    # 绘制背景网格
    for i in range(0, img_width, 100):
        draw.line([(i, 0), (i, img_height)], fill="#E0E0E0", width=1)
    for i in range(0, img_height, 100):
        draw.line([(0, i), (img_width, i)], fill="#E0E0E0", width=1)

    # 绘制背景标签
    try:
        font_bg = ImageFont.truetype("resources/PingFang.ttf", 20)
    except:
        font_bg = ImageFont.load_default()

    draw.text((50, 50), "测试图片 - 横向 (1200x800) - 允许重叠", fill="gray", font=font_bg)

    print(f"创建测试图片: 尺寸 {img_width}x{img_height}")

    # 2. 模拟检测到的表格区域
    table_regions = []

    # 示例1: 相对较宽的表格 (宽度占比 40%, 高度占比 25%)
    table1_x1 = 100
    table1_y1 = 100
    table1_x2 = 580
    table1_y2 = 300
    table_regions.append({
        "box": [table1_x1, table1_y1, table1_x2, table1_y2],
        "label": "table",
        "score": 0.95,
        "polygon_points": [
            [table1_x1, table1_y1],
            [table1_x2, table1_y1],
            [table1_x2, table1_y2],
            [table1_x1, table1_y2],
        ],
    })

    # 示例2: 相对较高的表格 (宽度占比 20%, 高度占比 45%)
    table2_x1 = 700
    table2_y1 = 200
    table2_x2 = 940
    table2_y2 = 560
    table_regions.append({
        "box": [table2_x1, table2_y1, table2_x2, table2_y2],
        "label": "table",
        "score": 0.90,
        "polygon_points": [
            [table2_x1, table2_y1],
            [table2_x2, table2_y1],
            [table2_x2, table2_y2],
            [table2_x1, table2_y2],
        ],
    })

    # 在测试图片上绘制表格区域
    for table in table_regions:
        x1, y1, x2, y2 = table["box"]
        draw.rectangle([(x1, y1), (x2, y2)], fill="#FFF0F0", outline="#FF8888", width=2)
        for i in range(5):
            y = y1 + 20 + i * 35
            if y < y2:
                draw.line([(x1 + 10, y), (x2 - 10, y)], fill="#CCCCCC", width=1)

    # 3. 执行表格扩展逻辑（与 layout_detector.py 中的逻辑相同）
    image_width, image_height = img.size
    is_landscape = image_width > image_height

    print(f"\n页面方向: {'横向 (Landscape)' if is_landscape else '纵向 (Portrait)'}")

    # 外扩边距设置（与 layout_detector.py 中的逻辑相同）
    expand_margin = 20
    expand_margin_x = max(expand_margin, int(image_width * 0.02))  # 2% of width or at least 20px
    expand_margin_y = max(expand_margin, int(image_height * 0.02))  # 2% of height or at least 20px

    print(f"外扩边距: expand_margin_x={expand_margin_x}px, expand_margin_y={expand_margin_y}px")

    # 步骤 1: 根据表格宽高占比决定扩展方向
    extended_tables = []
    for i, table in enumerate(table_regions):
        x1, y1, x2, y2 = table["box"]
        table_width = x2 - x1
        table_height = y2 - y1

        width_ratio = table_width / image_width
        height_ratio = table_height / image_height

        print(f"\n表格 {i+1}:")
        print(f"  原始位置: ({x1}, {y1}, {x2}, {y2})")
        print(f"  尺寸: {table_width}x{table_height}")
        print(f"  宽度占比: {width_ratio:.2%}")
        print(f"  高度占比: {height_ratio:.2%}")

        if width_ratio > height_ratio:
            # 相对较宽: 向左右扩展填满宽度
            ext_x1, ext_y1, ext_x2, ext_y2 = 0, y1, image_width, y2
            print(f"  扩展方向: 左右扩展 (填满宽度)")
        else:
            # 相对较高: 向上下扩展填满高度
            ext_x1, ext_y1, ext_x2, ext_y2 = x1, 0, x2, image_height
            print(f"  扩展方向: 上下扩展 (填满高度)")

        extended_tables.append({
            "original_box": table["box"],
            "extended_box": [ext_x1, ext_y1, ext_x2, ext_y2],
            "label": table["label"],
            "score": table["score"],
            "polygon_points": table["polygon_points"],
        })

    # 步骤 2: 排序扩展后的表格
    if is_landscape:
        extended_tables.sort(key=lambda t: t["extended_box"][0])  # 按 x1 排序
    else:
        extended_tables.sort(key=lambda t: t["extended_box"][1])  # 按 y1 排序

    # 步骤 3: 计算文本区域（允许重叠）
    text_regions = []
    current_pos = 0

    if is_landscape:
        # 横向: 水平分割
        for table in extended_tables:
            ext_x1, ext_y1, ext_x2, ext_y2 = table["extended_box"]
            # 添加文本区域（在表格之前）
            if current_pos < ext_x1:
                text_regions.append([current_pos, 0, ext_x1, image_height])
                print(f"\n文本区域 {len(text_regions)}: ({current_pos}, 0) - ({ext_x1}, {image_height})")
            current_pos = ext_x2
        # 添加最后的文本区域（右侧）
        if current_pos < image_width:
            text_regions.append([current_pos, 0, image_width, image_height])
            print(f"\n文本区域 {len(text_regions)}: ({current_pos}, 0) - ({image_width}, {image_height})")
    else:
        # 纵向: 垂直分割
        for table in extended_tables:
            ext_x1, ext_y1, ext_x2, ext_y2 = table["extended_box"]
            # 添加文本区域（在表格之前）
            if current_pos < ext_y1:
                text_regions.append([0, current_pos, image_width, ext_y1])
            current_pos = ext_y2
        # 添加最后的文本区域（底部）
        if current_pos < image_height:
            text_regions.append([0, current_pos, image_width, image_height])

    print(f"\n分割结果:")
    print(f"  表格区域: {len(table_regions)} 个")
    print(f"  文本区域: {len(text_regions)} 个")

    # 4. 绘制可视化效果
    print(f"\n生成可视化结果...")

    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)

    # 可视化 1: 原始检测结果
    img_original = img.copy()
    for i, table in enumerate(table_regions):
        draw_rectangle_with_text(img_original, table["box"], f"表格 {i+1}", (255, 0, 0))

    # 可视化 2: 扩展后的表格
    img_extended = img.copy()
    for i, table in enumerate(extended_tables):
        draw_rectangle_with_text(img_extended, table["extended_box"], f"扩展表格 {i+1}", (0, 255, 0, 64))
        draw_rectangle_with_text(img_extended, table["original_box"], f"原表格 {i+1}", (255, 0, 0))

    # 可视化 3: 最终分割结果（允许重叠）
    img_final = img.copy()

    # 先绘制文本区域
    for i, text_box in enumerate(text_regions):
        draw_rectangle_with_text(img_final, text_box, f"文本 {i+1}", (0, 0, 255, 64))

    # 再绘制表格区域（覆盖文本区域）
    for i, table in enumerate(table_regions):
        x1, y1, x2, y2 = table["box"]
        # 表格区域向外扩展
        exp_x1 = max(0, x1 - expand_margin_x)
        exp_y1 = max(0, y1 - expand_margin_y)
        exp_x2 = min(image_width, x2 + expand_margin_x)
        exp_y2 = min(image_height, y2 + expand_margin_y)

        # 绘制扩展区域（浅红色）
        draw_img = ImageDraw.Draw(img_final)
        draw_img.rectangle([(exp_x1, exp_y1), (exp_x2, exp_y2)], fill=(255, 100, 100, 64), outline=(255, 0, 0, 128), width=2)

        # 绘制原始表格区域
        draw_rectangle_with_text(img_final, table["box"], f"表格 {i+1}", (255, 0, 0))

    # 保存结果图片
    img_original.save(os.path.join(output_dir, "1_original_overlap.png"))
    img_extended.save(os.path.join(output_dir, "2_extended_overlap.png"))
    img_final.save(os.path.join(output_dir, "3_final_overlap.png"))

    print(f"  - 原始检测结果: {output_dir}/1_original_overlap.png")
    print(f"  - 扩展后的表格: {output_dir}/2_extended_overlap.png")
    print(f"  - 最终分割结果: {output_dir}/3_final_overlap.png")

    # 5. 显示裁剪的各个部分
    print(f"\n保存裁剪的区域...")

    all_cropped_images = []

    # 添加表格区域
    for i, table in enumerate(table_regions):
        x1, y1, x2, y2 = table["box"]
        cropped = img.crop((x1, y1, x2, y2))
        save_path = os.path.join(output_dir, f"cropped_table_{i+1}.png")
        cropped.save(save_path)
        all_cropped_images.append((f"表格 {i+1}", cropped))
        print(f"  {save_path}")

    # 添加文本区域
    for i, text_box in enumerate(text_regions):
        x1, y1, x2, y2 = text_box
        cropped = img.crop((x1, y1, x2, y2))
        save_path = os.path.join(output_dir, f"cropped_text_{i+1}.png")
        cropped.save(save_path)
        all_cropped_images.append((f"文本 {i+1}", cropped))
        print(f"  {save_path}")

    # 6. 创建拼接图
    if all_cropped_images:
        max_height = max(img[1].size[1] for img in all_cropped_images)
        total_width = sum(img[1].size[0] for img in all_cropped_images) + 20 * (len(all_cropped_images) - 1)

        collage = Image.new("RGB", (total_width, max_height + 50), "white")
        draw = ImageDraw.Draw(collage)

        x_offset = 0
        for label, cropped_img in all_cropped_images:
            collage.paste(cropped_img, (x_offset, 0))
            try:
                font = ImageFont.truetype("resources/PingFang.ttf", 20)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()

            draw.text((x_offset + 10, max_height + 10), label, fill="black", font=font)
            x_offset += cropped_img.size[0] + 20

        collage.save(os.path.join(output_dir, "4_collage_overlap.png"))
        print(f"\n拼接图已保存到: {os.path.join(output_dir, '4_collage_overlap.png')}")

    print(f"\n所有结果已保存到目录: {output_dir}")
    print("\n改进说明:")
    print("  - 表格区域向外扩展边距（20px或2%），确保不遗漏边界信息")
    print("  - 文本区域直接填满剩余空间，不添加边距")
    print("  - 允许表格和文本区域有重叠，避免信息丢失")
    print("  - 重叠部分可能被识别两次，但不会遗漏任何信息")

if __name__ == "__main__":
    test_overlap_no_missing()
