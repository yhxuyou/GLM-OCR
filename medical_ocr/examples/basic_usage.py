"""
Medical OCR 基础使用示例 - 使用 MedicalOcrPipeline
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 70)
    print("Medical OCR - 基础使用示例")
    print("=" * 70)
    print()

    # 1. 导入依赖
    print("1. 导入模块...")
    try:
        from medical_ocr import MedicalOcrPipeline
        from glmocr.config import load_config
        print("   ✅ 模块导入成功")
    except Exception as e:
        print(f"   ❌ 导入失败: {e}")
        return

    # 2. 加载配置
    print("\n2. 加载配置...")
    try:
        config = load_config()
        print("   ✅ 配置加载成功")
    except Exception as e:
        print(f"   ⚠️  配置加载失败: {e}")
        print("   提示: 请确保 glmocr 已安装并正确配置")
        return

    # 3. 创建 Pipeline
    print("\n3. 创建 MedicalOcrPipeline...")
    try:
        pipeline = MedicalOcrPipeline(
            config=config.pipeline,
            yolo_model_dir="/path/to/yolo/model",  # 替换为实际路径
            uvdoc_model_dir="/path/to/uvdoc/model",  # 替换为实际路径
        )
        print("   ✅ MedicalOcrPipeline 创建成功")
    except Exception as e:
        print(f"   ❌ Pipeline 创建失败: {e}")
        return

    # 4. 启动 Pipeline
    print("\n4. 启动 Pipeline...")
    try:
        pipeline.start()
        print("   ✅ Pipeline 启动成功")
    except Exception as e:
        print(f"   ⚠️  Pipeline 启动失败: {e}")
        print("   提示: 这通常是因为模型文件未找到，可以继续看示例用法")

    # 5. 使用说明
    print("\n" + "=" * 70)
    print("使用说明")
    print("=" * 70)
    print("""
MedicalOcrPipeline 继承自 glmocr.Pipeline，用法完全相同，只是增加了医疗文档预处理:

基本用法:
    # 1. 加载图片
    request_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "/path/to/medical/doc.jpg"}
                    }
                ]
            }
        ]
    }

    # 2. 处理（会自动应用预处理）
    for result in pipeline.process(request_data):
        print(result.json_result)
        print(result.markdown_result)

预处理流程 (自动执行):
    1. YOLO 文档检测 -> 裁剪
    2. RapidOCR 方向检测 -> 旋转
    3. UVDoc 畸变矫正 -> 平坦化

注意事项:
    - 首次调用会自动初始化模型（懒加载）
    - 无需修改现有代码，直接替换为 MedicalOcrPipeline 即可
    """)

    # 6. 停止
    print("\n6. 停止 Pipeline...")
    try:
        pipeline.stop()
        print("   ✅ Pipeline 已停止")
    except Exception as e:
        print(f"   ⚠️  停止失败: {e}")

    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
