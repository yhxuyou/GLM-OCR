#!/usr/bin/env python3
"""测试表格区域扩展和页面分割效果（包含边距）"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

# 确保能导入 glmocr 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'glmocr'))
sys.path.insert(0, os.path.dirname(__file__))

def draw_rectangle_with_text(img, box, label, color, text_color="white"):
    """在图片上绘制矩形框和文字"""
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=4)

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

def test_table_extension_with_margin():
    """测试表格区域扩展功能（包含边距）"""

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

    draw.text((50, 50), "测试图片 - 横向 (1200x800) - 带边距", fill="gray", font=font_bg)

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

    # 3. 执行表格扩展逻辑（与 layout_detector.py 中的逻辑相同，包含边距）
    image_width, image_height = img.size
    is_landscape = image_width > image_height

    print(f"\n页面方向: {'横向 (Landscape)' if is_landscape else '纵向 (Portrait)'}")

    # 边距设置（与 layout_detector.py 中的逻辑相同）
    margin = 10
    margin_x = max(margin, int(image_width * 0.01))  # 1% of width or at least 10px
    margin_y = max(margin, int(image_height * 0.01))  # 1% of height or at least 10px

    print(f"边距设置: margin_x={margin_x}px, margin_y={margin_y}px")

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

    # 步骤 3: 计算文本区域（添加边距）
    text_regions = []
    current_pos = 0

    if is_landscape:
        # 横向: 水平分割
        for table in extended_tables:
            ext_x1, ext_y1, ext_x2, ext_y2 = table["extended_box"]
            text_start = current_pos
            text_end = ext_x1

            # 只添加足够大的文本区域
            if text_end - text_start > margin_x * 2:
                # 在两边添加边距
                text_regions.append([
                    text_start + margin_x,
                    margin_y,
                    text_end - margin_x,
                    image_height - margin_y
                ])
                print(f"\n文本区域 {len(text_regions)}: ({text_start + margin_x}, {margin_y}) - ({text_end - margin_x}, {image_height - margin_y})")
            elif text_end > text_start:
                text_regions.append([text_start, 0, text_end, image_height])
                print(f"\n文本区域 {len(text_regions)}: ({text_start}, 0) - ({text_end}, {image_height})")

            current_pos = ext_x2

        # 添加最后的文本区域（右侧）
        if image_width - current_pos > margin_x * 2:
            text_regions.append([
                current_pos + margin_x,
                margin_y,
                image_width - margin_x,
                image_height - margin_y
            ])
            print(f"\n文本区域 {len(text_regions)}: ({current_pos + margin_x}, {margin_y}) - ({image_width - margin_x}, {image_height - margin_y})")
        elif image_width > current_pos:
            text_regions.append([current_pos, 0, image_width, image_height])
    else:
        # 纵向: 垂直分割
        for table in extended_tables:
            ext_x1, ext_y1, ext_x2, ext_y2 = table["extended_box"]
            text_start = current_pos
            text_end = ext_y1

            # 只添加足够大的文本区域
            if text_end - text_start > margin_y * 2:
                # 在两边添加边距
                text_regions.append([
                    margin_x,
                    text_start + margin_y,
                    image_width - margin_x,
                    text_end - margin_y
                ])
            elif text_end > text_start:
                text_regions.append([0, text_start, image_width, text_end])

            current_pos = ext_y2

        # 添加最后的文本区域（底部）
        if image_height - current_pos > margin_y * 2:
            text_regions.append([
                margin_x,
                current_pos + margin_y,
                image_width - margin_x,
                image_height - margin_y
            ])
        elif image_height > current_pos:
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
        draw_rectangle_with_text(img_extended, table["extended_box"], f"扩展表格 {i+1}", (0, 255, 0))
        draw_rectangle_with_text(img_extended, table["original_box"], f"原表格 {i+1}", (255, 0, 0))

    # 可视化 3: 最终分割结果（带边距）
    img_final = img.copy()

    # 先绘制文本区域（带边距）
    for i, text_box in enumerate(text_regions):
        x1, y1, x2, y2 = text_box
        draw_img = ImageDraw.Draw(img_final)
        draw_img.rectangle([(x1, y1), (x2, y2)], fill="#F0F8FF")
        draw_rectangle_with_text(img_final, text_box, f"文本 {i+1}", (0, 0, 255))

    # 再绘制表格区域
    for i, table in enumerate(table_regions):
        x1, y1, x2, y2 = table["box"]
        draw_img = ImageDraw.Draw(img_final)
        draw_img.rectangle([(x1, y1), (x2, y2)], fill="#FFF5F5")
        draw_rectangle_with_text(img_final, table["box"], f"表格 {i+1}", (255, 0, 0))

    # 保存结果图片
    img_original.save(os.path.join(output_dir, "1_original_with_margin.png"))
    img_extended.save(os.path.join(output_dir, "2_extended_with_margin.png"))
    img_final.save(os.path.join(output_dir, "3_final_with_margin.png"))

    print(f"  - 原始检测结果: {output_dir}/1_original_with_margin.png")
    print(f"  - 扩展后的表格: {output_dir}/2_extended_with_margin.png")
    print(f"  - 最终分割结果: {output_dir}/3_final_with_margin.png")

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

        collage.save(os.path.join(output_dir, "4_collage_with_margin.png"))
        print(f"\n拼接图已保存到: {os.path.join(output_dir, '4_collage_with_margin.png')}")

    print(f"\n所有结果已保存到目录: {output_dir}")
    print("\n改进说明:")
    print("  - 文本区域周围添加了边距，避免与表格区域紧密贴合")
    print("  - 边距大小: 至少10像素或图片宽高的1%")
    print("  - 这样可以避免关键信息丢失，特别是靠近边界的文字")

if __name__ == "__main__":
    test_table_extension_with_margin()
