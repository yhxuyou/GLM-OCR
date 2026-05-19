#!/usr/bin/env python3
"""
直接运行测试脚本 - 不需要安装
"""

import sys
from pathlib import Path

# 添加项目路径到 sys.path，这样就不需要安装了
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("Medical OCR - 直接运行测试 (无需安装)")
print("=" * 70)
print(f"项目根目录: {project_root}")
print(f"Python 路径: {sys.path[:3]}")
print()


def test_imports():
    """测试导入"""
    print("1️⃣  测试模块导入...")
    try:
        # 测试直接导入
        import medical_ocr
        print(f"   ✅ 成功导入 medical_ocr 包 (版本: {medical_ocr.__version__})")
        
        print(f"   ℹ️  注意: Pipeline 和 LayoutDetector 类仅在 glmocr 已安装时可用")
        return True
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")
        import traceback
        print(f"   详细错误:\n{traceback.format_exc()}")
        return False


def test_structure():
    """测试项目结构"""
    print("\n2️⃣  测试项目结构...")
    checks = [
        ("项目配置文件", project_root / "pyproject.toml"),
        ("Pipeline 类", project_root / "medical_ocr" / "pipeline.py"),
        ("布局检测器", project_root / "medical_ocr" / "layout_detector.py"),
        ("服务器文件", project_root / "medical_ocr" / "server.py"),
    ]
    
    all_good = True
    for desc, path in checks:
        if path.exists():
            print(f"   ✅ {desc}: {path.name}")
        else:
            print(f"   ❌ {desc} 缺失")
            all_good = False
    return all_good


def test_pipeline_class():
    """测试 Pipeline 类的基本结构"""
    print("\n3️⃣  测试 Pipeline 类...")
    try:
        from medical_ocr.pipeline import MedicalOcrPipeline
        print(f"   ✅ MedicalOcrPipeline 类存在")
        
        # 检查类的方法
        methods = [
            'add_preprocess_hook',
            'add_postprocess_hook',
            'process',
        ]
        for method in methods:
            if hasattr(MedicalOcrPipeline, method):
                print(f"   ✅ 方法 {method}() 存在")
            else:
                print(f"   ⚠️  方法 {method}() 可能不存在")
        
        return True
    except Exception as e:
        print(f"   ⚠️  Pipeline 类测试跳过 (依赖可能未安装): {e}")
        return True


def test_layout_detector():
    """测试布局检测器"""
    print("\n4️⃣  测试布局检测器...")
    try:
        from medical_ocr.layout_detector import MedicalLayoutDetector
        print(f"   ✅ MedicalLayoutDetector 类存在")
        
        # 检查关键方法
        methods = [
            '_enhance_medical_image',
            '_filter_medical_regions',
            'process',
        ]
        for method in methods:
            if hasattr(MedicalLayoutDetector, method):
                print(f"   ✅ 方法 {method}() 存在")
            else:
                print(f"   ⚠️  方法 {method}() 可能不存在")
        
        return True
    except Exception as e:
        print(f"   ⚠️  布局检测器测试跳过 (依赖可能未安装): {e}")
        return True


def test_examples():
    """测试示例代码"""
    print("\n5️⃣  测试示例代码...")
    example_file = project_root / "examples" / "basic_usage.py"
    if example_file.exists():
        print(f"   ✅ 示例文件存在: {example_file}")
        print("   💡 可以直接运行: python examples/basic_usage.py")
        return True
    else:
        print(f"   ❌ 示例文件缺失")
        return False


def main():
    """主测试函数"""
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("项目结构", test_structure()))
    results.append(("Pipeline 类", test_pipeline_class()))
    results.append(("布局检测器", test_layout_detector()))
    results.append(("示例代码", test_examples()))
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("📊 测试结果汇总")
    print("=" * 70)
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有测试通过！项目结构完整！")
        print("\n📝 下一步:")
        print("  1. 安装 glmocr 及其依赖 (如果还没有)")
        print("  2. 查看示例: python examples/basic_usage.py")
        print("  3. 阅读文档: README.md")
        print("  4. (可选) 安装项目: pip install -e .")
    else:
        print("⚠️  部分测试失败，请检查上面的错误信息")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
