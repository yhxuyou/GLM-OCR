#!/usr/bin/env python3
"""
医疗费用清单处理器测试脚本
"""

import sys
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image

from deploy.medical_expense_processor import (
    MedicalExpenseProcessor,
    MedicalExpenseResult,
    create_medical_expense_processor
)


def test_basic_processing():
    """测试基本处理功能"""
    print("=" * 60)
    print("测试1: 基本处理功能")
    print("=" * 60)
    
    # 创建测试图像
    img = Image.new('RGB', (1000, 800), color='white')
    
    # 模拟布局检测结果
    layout_results = [
        {
            "label": "table",
            "bbox_2d": [100, 200, 500, 600],
            "score": 0.95,
            "polygon": [[100, 200], [500, 200], [500, 600], [100, 600]]
        },
        {
            "label": "text",
            "bbox_2d": [50, 50, 950, 150],
            "score": 0.90,
            "polygon": [[50, 50], [950, 50], [950, 150], [50, 150]]
        },
        {
            "label": "text",
            "bbox_2d": [550, 100, 900, 180],
            "score": 0.88,
            "polygon": [[550, 100], [900, 100], [900, 180], [550, 180]]
        }
    ]
    
    # 处理
    processor = create_medical_expense_processor(
        table_expand_ratio=0.05,
        table_min_expansion=10
    )
    
    result = processor.process(img, layout_results)
    
    # 验证结果
    print(f"✓ 检测到 {len(result.table_regions)} 个表格区域")
    print(f"✓ 非表格区域数量: {result.non_table_image.cropped_regions_count if result.non_table_image else 0}")
    
    # 检查表格扩展
    if result.table_regions:
        table = result.table_regions[0]
        print(f"✓ 表格原始bbox: {table.bbox}")
        print(f"✓ 表格扩展bbox: {table.expanded_bbox}")
        
        # 计算扩展量
        orig_width = table.bbox[2] - table.bbox[0]
        orig_height = table.bbox[3] - table.bbox[1]
        exp_width = table.expanded_bbox[2] - table.expanded_bbox[0]
        exp_height = table.expanded_bbox[3] - table.expanded_bbox[1]
        
        print(f"✓ 宽度扩展: {orig_width} -> {exp_width} (+{(exp_width/orig_width-1)*100:.1f}%)")
        print(f"✓ 高度扩展: {orig_height} -> {exp_height} (+{(exp_height/orig_height-1)*100:.1f}%)")
    
    # 保存测试结果
    output_dir = project_root / "deploy" / "test_output"
    saved_files = result.save(str(output_dir))
    print(f"✓ 结果已保存到: {output_dir}")
    print(f"  保存的文件: {list(saved_files.keys())}")
    
    print("\n基本处理测试通过!\n")
    return True


def test_table_expansion():
    """测试表格扩展比例"""
    print("=" * 60)
    print("测试2: 表格扩展比例")
    print("=" * 60)
    
    img = Image.new('RGB', (2000, 1500), color='white')
    
    # 测试不同的扩展比例
    test_cases = [
        (0.05, 10, "5%扩展，最小10px"),
        (0.1, 20, "10%扩展，最小20px"),
        (0.02, 50, "2%扩展，最小50px"),
    ]
    
    layout_results = [
        {
            "label": "table",
            "bbox_2d": [500, 500, 1500, 1000],  # 1000x500像素
            "score": 0.95
        }
    ]
    
    for ratio, min_exp, desc in test_cases:
        processor = MedicalExpenseProcessor(
            table_expand_ratio=ratio,
            table_min_expansion=min_exp
        )
        
        result = processor.process(img, layout_results)
        
        if result.table_regions:
            table = result.table_regions[0]
            orig_width = table.bbox[2] - table.bbox[0]
            orig_height = table.bbox[3] - table.bbox[1]
            exp_width = table.expanded_bbox[2] - table.expanded_bbox[0]
            exp_height = table.expanded_bbox[3] - table.expanded_bbox[1]
            
            print(f"\n{desc}:")
            print(f"  原始: {orig_width}x{orig_height}")
            print(f"  扩展: {exp_width}x{exp_height}")
            print(f"  实际扩展: 宽度+{(exp_width/orig_width-1)*100:.1f}%, 高度+{(exp_height/orig_height-1)*100:.1f}%")
    
    print("\n表格扩展测试完成!\n")
    return True


def test_non_table_merging():
    """测试非表格区域合并"""
    print("=" * 60)
    print("测试3: 非表格区域合并")
    print("=" * 60)
    
    # 创建包含多个区域的测试图像
    img = Image.new('RGB', (1000, 800), color='white')
    
    layout_results = [
        {"label": "table", "bbox_2d": [400, 300, 600, 500], "score": 0.95},
        {"label": "text", "bbox_2d": [50, 50, 350, 150], "score": 0.90},
        {"label": "title", "bbox_2d": [50, 200, 950, 280], "score": 0.88},
        {"label": "footer", "bbox_2d": [50, 600, 950, 750], "score": 0.85},
    ]
    
    processor = MedicalExpenseProcessor()
    result = processor.process(img, layout_results)
    
    print(f"✓ 检测到 {len(result.table_regions)} 个表格")
    print(f"✓ 合并后的非表格图像尺寸: {result.non_table_image.merged_image.size}")
    print(f"✓ 原始图像尺寸: {result.image_size}")
    
    # 验证图像尺寸一致
    assert result.non_table_image.merged_image.size == result.image_size
    print("✓ 合并图像尺寸与原始图像一致")
    
    print("\n非表格合并测试通过!\n")
    return True


def test_save_and_load():
    """测试保存和加载功能"""
    print("=" * 60)
    print("测试4: 保存和加载功能")
    print("=" * 60)
    
    img = Image.new('RGB', (1000, 800), color='white')
    
    layout_results = [
        {
            "label": "table",
            "bbox_2d": [100, 100, 500, 400],
            "score": 0.95
        }
    ]
    
    processor = MedicalExpenseProcessor()
    result = processor.process(img, layout_results)
    
    # 保存
    output_dir = project_root / "deploy" / "test_save"
    saved = result.save(str(output_dir))
    
    print(f"✓ 保存的文件:")
    for name, path in saved.items():
        print(f"  - {name}: {path}")
    
    # 验证文件存在
    for path in saved.values():
        assert Path(path).exists(), f"文件不存在: {path}"
    
    print("✓ 所有保存的文件都存在")
    print("\n保存功能测试通过!\n")
    return True


def test_to_dict():
    """测试字典转换"""
    print("=" * 60)
    print("测试5: 字典转换")
    print("=" * 60)
    
    img = Image.new('RGB', (1000, 800), color='white')
    
    layout_results = [
        {"label": "table", "bbox_2d": [100, 100, 500, 400], "score": 0.95},
        {"label": "text", "bbox_2d": [50, 50, 950, 80], "score": 0.90}
    ]
    
    processor = MedicalExpenseProcessor()
    result = processor.process(img, layout_results)
    
    # 转换为字典
    result_dict = result.to_dict()
    
    print(f"✓ 结果字典包含:")
    print(json.dumps(result_dict, indent=2, ensure_ascii=False))
    
    print("\n字典转换测试通过!\n")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("=" * 60)
    print("  医疗费用清单处理器测试套件")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_processing,
        test_table_expansion,
        test_non_table_merging,
        test_save_and_load,
        test_to_dict,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test.__name__} 失败")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} 出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"  测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    print()
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
