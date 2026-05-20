"""
Medical OCR 基础使用示例 - 使用 MedicalPageLoader
"""

import sys
from pathlib import Path
from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 70)
    print("Medical OCR - MedicalPageLoader 使用示例")
    print("=" * 70)

    # 1. 导入 MedicalPageLoader
    print("\n1️⃣  导入模块...")
    try:
        from medical_ocr import MedicalPageLoader
        from glmocr.config import PageLoaderConfig
        print("   ✅ 模块导入成功")
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")
        return

    # 2. 创建配置
    print("\n2️⃣  创建配置...")
    try:
        config = PageLoaderConfig()
        print("   ✅ PageLoaderConfig 创建成功")
    except Exception as e:
        print(f"   ❌ 配置创建失败: {e}")
        print("   ℹ️  使用默认配置参数")
        config = None

    # 3. 创建 MedicalPageLoader
    print("\n3️⃣  创建 MedicalPageLoader...")
    try:
        if config:
            page_loader = MedicalPageLoader(
                config=config,
                yolo_model_dir="/path/to/yolo/model",      # YOLO 模型路径
                uvdoc_model_dir="/path/to/uvdoc/model",   # UVDoc 模型路径
                enable_preprocessing=True
            )
        else:
            # 如果没有配置，创建最小化实例
            from glmocr.dataloader import PageLoader
            page_loader = MedicalPageLoader.__new__(MedicalPageLoader)
            page_loader.config = None
            page_loader.enable_preprocessing = True
            page_loader.yolo_model_dir = None
            page_loader.uvdoc_model_dir = None
            page_loader._yolo_model = None
            page_loader._uvdoc_model = None
            page_loader._rapidocr_detector = None

        print("   ✅ MedicalPageLoader 创建成功")
    except Exception as e:
        print(f"   ❌ MedicalPageLoader 创建失败: {e}")
        return

    # 4. 展示预处理功能
    print("\n4️⃣  预处理功能说明...")
    print("   MedicalPageLoader 提供三阶段预处理：")
    print("   ├─ 1. YOLO 文档检测 - 检测文档区域并裁剪")
    print("   ├─ 2. RapidOCR 方向检测 - 检测并矫正文档方向")
    print("   └─ 3. UVDoc 扭曲矫正 - 矫正文档畸变")

    # 5. 预处理流程说明
    print("\n5️⃣  预处理流程...")
    print("   preprocess_image() 方法流程：")
    print("   1. detect_document(image)      - 使用 YOLO 检测文档")
    print("   2. crop_document(image, bbox)  - 裁剪文档区域")
    print("   3. correct_orientation(image)  - 使用 RapidOCR 矫正方向")
    print("   4. correct_distortion(image)   - 使用 UVDoc 矫正畸变")

    # 6. 代码示例
    print("\n6️⃣  代码示例...")
    print("""
    # 创建并初始化
    page_loader = MedicalPageLoader(config)
    page_loader.start()

    # 加载并预处理图片
    pages = page_loader.load_pages(["document.jpg"])
    # pages 现在包含经过完整预处理的图片

    # 或者手动预处理单张图片
    from PIL import Image
    image = Image.open("medical_doc.jpg")
    processed = page_loader.preprocess_image(image)

    # 完成后停止
    page_loader.stop()
    """)

    # 7. API 说明
    print("\n7️⃣  主要 API 方法...")
    methods = [
        ("start()", "初始化所有模型"),
        ("stop()", "释放模型资源"),
        ("load_pages(sources)", "加载并预处理页面"),
        ("preprocess_image(image)", "预处理单张图片"),
        ("detect_document(image)", "YOLO 文档检测"),
        ("correct_orientation(image)", "RapidOCR 方向矫正"),
        ("correct_distortion(image)", "UVDoc 扭曲矫正"),
    ]
    for method, desc in methods:
        print(f"   • {method:<35} - {desc}")

    print("\n" + "=" * 70)
    print("✅ 示例完成！")
    print("\n📝 注意事项：")
    print("   1. 需要安装依赖：pip install ultralytics rapidocr_onnxruntime")
    print("   2. 需要提供 YOLO 和 UVDoc 模型路径")
    print("   3. 查看文档了解详细的配置选项")
    print("=" * 70)


if __name__ == "__main__":
    main()
