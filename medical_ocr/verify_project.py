"""
验证 medical-ocr 项目结构
"""

import sys
from pathlib import Path

def check_file(file_path: Path, description: str) -> bool:
    """检查文件是否存在"""
    if file_path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} 缺失: {file_path}")
        return False

def main():
    print("=" * 60)
    print("Medical OCR 项目结构验证")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    
    # 检查主要文件
    checks = [
        (project_root / "pyproject.toml", "项目配置文件"),
        (project_root / "README.md", "项目说明文档"),
        (project_root / "medical_ocr" / "__init__.py", "包初始化文件"),
        (project_root / "medical_ocr" / "pipeline.py", "Pipeline 类"),
        (project_root / "medical_ocr" / "layout_detector.py", "布局检测器"),
        (project_root / "medical_ocr" / "server.py", "服务器文件"),
        (project_root / "examples" / "basic_usage.py", "示例代码"),
        (project_root / "tests" / "__init__.py", "测试文件"),
    ]
    
    print("\n📁 文件检查:")
    all_good = True
    for file_path, description in checks:
        if not check_file(file_path, description):
            all_good = False
    
    # 检查目录结构
    print("\n📂 目录结构:")
    expected_dirs = ["medical_ocr", "examples", "tests"]
    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        if dir_path.is_dir():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/")
            all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("✅ 项目结构验证通过！")
        print("\n快速开始:")
        print("  1. 安装依赖: pip install -e .")
        print("  2. 查看示例: python examples/basic_usage.py")
        print("  3. 阅读文档: README.md")
    else:
        print("❌ 项目结构有问题，请检查以上错误！")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
