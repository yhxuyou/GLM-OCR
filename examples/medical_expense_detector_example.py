#!/usr/bin/env python3
"""
Medical Expense Layout Detector - 医疗费用清单检测器使用示例

演示如何使用 MedicalExpenseLayoutDetector 处理医疗费用清单图像。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
import argparse

from glmocr.config import load_config
from glmocr.layout.medical_detector import (
    MedicalExpenseLayoutDetector,
    create_medical_expense_detector,
)


def main():
    parser = argparse.ArgumentParser(
        description="医疗费用清单布局检测器使用示例"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入图像路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./medical_output",
        help="输出目录路径",
    )
    parser.add_argument(
        "--expand-ratio",
        type=float,
        default=0.05,
        help="表格扩展比例 (默认: 0.05)",
    )
    parser.add_argument(
        "--min-expansion",
        type=int,
        default=10,
        help="最小扩展像素 (默认: 10)",
    )
    args = parser.parse_args()

    # 1. 加载配置
    print("正在加载配置...")
    config = load_config(args.config)

    # 2. 创建医疗费用清单检测器
    print("正在创建医疗费用清单检测器...")
    detector = create_medical_expense_detector(
        config.layout,
        table_expand_ratio=args.expand_ratio,
        table_min_expansion=args.min_expansion,
    )

    # 3. 加载图像
    print(f"正在加载图像: {args.image}")
    image = Image.open(args.image)

    # 4. 启动检测器
    print("正在启动检测器...")
    detector.start()

    try:
        # 5. 处理图像
        print("正在处理图像...")
        results, vis_images, medical_results = detector.process(
            images=[image],
            save_visualization=True,
            return_medical_result=True,
        )

        # 6. 获取医疗费用清单结果
        medical_result = medical_results[0]
        print(f"\n处理完成！")
        print(f"  检测到 {len(medical_result.table_regions)} 个表格")
        for table in medical_result.table_regions:
            print(f"    表格 {table.table_id}:")
            print(f"      原始坐标: {table.bbox_pixel}")
            print(f"      扩展坐标: {table.expanded_bbox_pixel}")

        if medical_result.non_table_image:
            print(f"  非表格整体图尺寸: {medical_result.non_table_image.merged_image.size}")

        # 7. 保存结果
        print(f"\n正在保存结果到: {args.output}")
        saved_files = medical_result.save(args.output)
        print(f"  已保存文件:")
        for name, path in saved_files.items():
            print(f"    - {name}: {path}")

        # 8. 保存可视化
        if vis_images:
            vis_path = Path(args.output) / "layout_visualization.png"
            vis_images[0].save(vis_path)
            print(f"    - layout_visualization: {vis_path}")

    finally:
        # 9. 停止检测器
        print("正在停止检测器...")
        detector.stop()

    print("\n所有任务完成！")


def simple_example():
    """简单使用示例"""
    print("\n" + "="*60)
    print("简单使用示例")
    print("="*60)

    from PIL import Image
    from glmocr.config import load_config
    from glmocr.layout.medical_detector import create_medical_expense_detector

    # 1. 加载配置
    config = load_config()

    # 2. 创建检测器
    detector = create_medical_expense_detector(
        config.layout,
        table_expand_ratio=0.05,
        table_min_expansion=10,
        table_labels=["table", "表格"],
    )

    # 3. 启动
    detector.start()

    # 4. 加载图像
    image = Image.open("medical_expense.jpg")

    # 5. 处理
    results, vis, medical_results = detector.process(
        [image],
        save_visualization=True,
        return_medical_result=True,
    )

    # 6. 获取结果
    medical_result = medical_results[0]

    # 7. 遍历表格
    for table in medical_result.table_regions:
        # 保存表格裁剪图
        table.cropped_image.save(f"table_{table.table_id}.png")

    # 8. 保存非表格整体图
    if medical_result.non_table_image:
        medical_result.non_table_image.merged_image.save("non_table.png")

    # 9. 全部保存
    medical_result.save("./output")

    # 10. 停止
    detector.stop()


if __name__ == "__main__":
    main()
