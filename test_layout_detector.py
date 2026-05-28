#!/usr/bin/env python3
"""直接测试 layout_detector.py 的表格扩展功能"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

# 添加 glmocr 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'glmocr'))

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

def test_table_extension_logic():
    """测试表格扩展逻辑（不依赖模型）"""
    print("=" * 60)
    print("测试 layout_detector.py 的表格扩展逻辑")
    print("=" * 60)

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

    draw.text((50, 50), "测试图片 - 横向 (1200x800)", fill="gray", font=font_bg)

    print(f"\n1. 创建测试图片: 尺寸 {img_width}x{img_height}")

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

    print(f"\n2. 页面方向: {'横向 (Landscape)' if is_landscape else '纵向 (Portrait)'}")

    # 步骤 1: 根据表格宽高占比决定扩展方向
    extended_tables = []
    for i, table in enumerate(table_regions):
        x1, y1, x2, y2 = table["box"]
        table_width = x2 - x1
        table_height = y2 - y1

        width_ratio = table_width / image_width
        height_ratio = table_height / image_height

        print(f"\n3. 表格 {i+1}:")
        print(f"   - 原始位置: ({x1}, {y1}, {x2}, {y2})")
        print(f"   - 尺寸: {table_width}x{table_height}")
        print(f"   - 宽度占比: {width_ratio:.2%}")
        print(f"   - 高度占比: {height_ratio:.2%}")

        if width_ratio > height_ratio:
            # 相对较宽: 向左右扩展填满宽度
            ext_x1, ext_y1, ext_x2, ext_y2 = 0, y1, image_width, y2
            print(f"   - 扩展方向: 左右扩展 (填满宽度)")
        else:
            # 相对较高: 向上下扩展填满高度
            ext_x1, ext_y1, ext_x2, ext_y2 = x1, 0, x2, image_height
            print(f"   - 扩展方向: 上下扩展 (填满高度)")

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

    # 步骤 3: 计算文本区域
    text_regions = []
    current_pos = 0

    if is_landscape:
        # 横向: 水平分割
        for table in extended_tables:
            ext_x1, ext_y1, ext_x2, ext_y2 = table["extended_box"]
            if current_pos < ext_x1:
                text_regions.append([current_pos, 0, ext_x1, image_height])
            current_pos = ext_x2
        if current_pos < image_width:
            text_regions.append([current_pos, 0, image_width, image_height])
    else:
        # 纵向: 垂直分割
        for table in extended_tables:
            ext_x1, ext_y1, ext_x2, ext_y2 = table["extended_box"]
            if current_pos < ext_y1:
                text_regions.append([0, current_pos, image_width, ext_y1])
            current_pos = ext_y2
        if current_pos < image_height:
            text_regions.append([0, current_pos, image_width, image_height])

    print(f"\n4. 分割结果:")
    print(f"   - 表格区域: {len(table_regions)} 个")
    print(f"   - 文本区域: {len(text_regions)} 个")

    # 4. 绘制可视化效果
    print(f"\n5. 生成可视化结果...")

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

    # 可视化 3: 最终分割结果
    img_final = img.copy()

    # 先绘制文本区域
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
    img_original.save(os.path.join(output_dir, "1_original_detection.png"))
    img_extended.save(os.path.join(output_dir, "2_extended_tables.png"))
    img_final.save(os.path.join(output_dir, "3_final_result.png"))

    print(f"   - 原始检测结果: {output_dir}/1_original_detection.png")
    print(f"   - 扩展后的表格: {output_dir}/2_extended_tables.png")
    print(f"   - 最终分割结果: {output_dir}/3_final_result.png")

    # 5. 保存裁剪的区域
    print(f"\n6. 保存裁剪的区域...")

    all_cropped_images = []

    for i, table in enumerate(table_regions):
        x1, y1, x2, y2 = table["box"]
        cropped = img.crop((x1, y1, x2, y2))
        save_path = os.path.join(output_dir, f"cropped_table_{i+1}.png")
        cropped.save(save_path)
        all_cropped_images.append((f"表格 {i+1}", cropped))
        print(f"   - {save_path}")

    for i, text_box in enumerate(text_regions):
        x1, y1, x2, y2 = text_box
        cropped = img.crop((x1, y1, x2, y2))
        save_path = os.path.join(output_dir, f"cropped_text_{i+1}.png")
        cropped.save(save_path)
        all_cropped_images.append((f"文本 {i+1}", cropped))
        print(f"   - {save_path}")

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

        collage.save(os.path.join(output_dir, "4_collage.png"))
        print(f"   - 拼接图: {output_dir}/4_collage.png")

    print("\n7. 测试完成！")
    print(f"   所有结果已保存到 {output_dir} 目录")

def test_with_real_detector():
    """测试真实的布局检测器"""
    print("\n" + "=" * 60)
    print("测试真实的 layout_detector")
    print("=" * 60)

    try:
        from glmocr.layout.layout_detector import PPDocLayoutDetector
        from glmocr.config import GlmOcrConfig
    except ImportError as e:
        print(f"无法导入必要的模块: {e}")
        print("提示: 请安装所需的依赖包")
        return

    # 检查模型是否存在
    model_dir = "/workspace/models/PP-DocLayoutV3_safetensors"
    if not os.path.exists(model_dir):
        print(f"\n警告: 模型目录不存在: {model_dir}")
        print("跳过真实检测器测试")
        print("\n提示: 如果要测试真实的布局检测器，请下载模型:")
        print("  https://hf-mirror.com/PaddlePaddle/PP-DocLayoutV3_safetensors")
        return

    # 1. 加载测试图片
    test_image_path = "examples/source/table.png"
    if not os.path.exists(test_image_path):
        print(f"\n错误: 测试图片 {test_image_path} 不存在！")
        return

    img = Image.open(test_image_path).convert("RGB")
    img_width, img_height = img.size
    print(f"\n1. 加载图片: {test_image_path}")
    print(f"   尺寸: {img_width}x{img_height}")

    # 2. 初始化布局检测器
    print("\n2. 初始化布局检测器...")

    try:
        config = GlmOcrConfig.from_yaml("test_config.yaml")
        layout_detector = PPDocLayoutDetector(config.pipeline.layout)
        print("   布局检测器创建成功")
    except Exception as e:
        print(f"   配置加载失败: {e}")
        return

    # 3. 启动模型
    print("\n3. 启动模型...")
    try:
        layout_detector.start()
        print("   模型启动成功")
    except Exception as e:
        print(f"   模型启动失败: {e}")
        return

    # 4. 执行检测
    print("\n4. 执行布局检测...")
    try:
        results, vis_images = layout_detector.process(
            images=[img],
            save_visualization=True,
            global_start_idx=0,
            use_polygon=False
        )

        print(f"   检测完成！")
        print(f"   检测到 {len(results[0])} 个区域")

        # 5. 分析结果
        print("\n5. 检测结果分析:")
        table_count = 0
        text_count = 0

        for i, region in enumerate(results[0]):
            label = region.get('label', 'unknown')
            bbox = region.get('bbox_2d', [])
            task_type = region.get('task_type', 'unknown')

            print(f"   区域 {i+1}:")
            print(f"     - 标签: {label}")
            print(f"     - 任务类型: {task_type}")
            print(f"     - 边界框: {bbox}")
            print(f"     - 置信度: {region.get('score', 0):.2f}")

            if task_type == 'table':
                table_count += 1
            elif task_type == 'text':
                text_count += 1

        print(f"\n   统计:")
        print(f"     - 表格区域: {table_count} 个")
        print(f"     - 文本区域: {text_count} 个")

        # 6. 可视化结果
        print("\n6. 生成可视化结果...")

        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)

        # 可视化 1: 使用 layout_detector 自带可视化
        if vis_images:
            for idx, vis_img in vis_images.items():
                vis_img.save(os.path.join(output_dir, f"layout_visualization_page{idx}.png"))
            print(f"   - 布局可视化已保存到 {output_dir}/layout_visualization_page0.png")

        # 可视化 2: 自定义可视化
        img_result = img.copy()
        draw = ImageDraw.Draw(img_result)

        for i, region in enumerate(results[0]):
            label = region.get('label', 'unknown')
            bbox = region.get('bbox_2d', [])
            task_type = region.get('task_type', 'unknown')

            # 反归一化坐标
            x1 = int(bbox[0] / 1000 * img_width)
            y1 = int(bbox[1] / 1000 * img_height)
            x2 = int(bbox[2] / 1000 * img_width)
            y2 = int(bbox[3] / 1000 * img_height)

            if task_type == 'table':
                color = (255, 0, 0)  # 红色
                draw_rectangle_with_text(img_result, [x1, y1, x2, y2], f"表格 {i+1}", color)
            elif task_type == 'text':
                color = (0, 0, 255)  # 蓝色
                draw_rectangle_with_text(img_result, [x1, y1, x2, y2], f"文本 {i+1}", color)

        img_result.save(os.path.join(output_dir, "detection_result_custom.png"))
        print(f"   - 自定义可视化已保存到 {output_dir}/detection_result_custom.png")

        # 7. 保存裁剪的区域
        print("\n7. 保存裁剪的区域...")

        for i, region in enumerate(results[0]):
            label = region.get('label', 'unknown')
            bbox = region.get('bbox_2d', [])
            task_type = region.get('task_type', 'unknown')

            x1 = int(bbox[0] / 1000 * img_width)
            y1 = int(bbox[1] / 1000 * img_height)
            x2 = int(bbox[2] / 1000 * img_width)
            y2 = int(bbox[3] / 1000 * img_height)

            cropped = img.crop((x1, y1, x2, y2))
            cropped_path = os.path.join(output_dir, f"cropped_{task_type}_{i+1}.png")
            cropped.save(cropped_path)
            print(f"   - {cropped_path}")

        print("\n8. 测试完成！")
        print(f"   所有结果已保存到 {output_dir} 目录")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 9. 停止模型
        print("\n9. 停止模型...")
        layout_detector.stop()
        print("   模型已停止")

if __name__ == "__main__":
    # 先测试表格扩展逻辑
    test_table_extension_logic()

    # 再测试真实的检测器
    test_with_real_detector()
