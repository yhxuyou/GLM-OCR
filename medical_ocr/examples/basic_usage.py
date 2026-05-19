"""
Medical OCR 基础使用示例
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from glmocr.config import load_config
from medical_ocr.pipeline import MedicalOcrPipeline
from medical_ocr.layout_detector import MedicalLayoutDetector


def main():
    print("Medical OCR 基础示例")
    print("=" * 50)
    
    # 1. 加载配置
    print("加载配置...")
    config = load_config()
    print(f"配置加载完成: {config}")
    
    # 2. 创建医疗专用布局检测器
    print("\n创建医疗布局检测器...")
    layout_detector = MedicalLayoutDetector(config.pipeline.layout)
    print("医疗布局检测器创建完成")
    
    # 3. 创建医疗 OCR pipeline
    print("\n创建医疗 OCR pipeline...")
    pipeline = MedicalOcrPipeline(
        config=config.pipeline,
        layout_detector=layout_detector
    )
    print("医疗 OCR pipeline 创建完成")
    
    # 4. 添加自定义钩子示例
    print("\n添加自定义处理钩子...")
    
    def sample_preprocess(image, context):
        """示例预处理钩子"""
        print(f"  预处理: 图像尺寸 = {image.size}")
        return image
    
    def sample_postprocess(json_result, markdown_result, context):
        """示例后处理钩子"""
        print(f"  后处理: JSON长度 = {len(json_result)}, Markdown长度 = {len(markdown_result)}")
        return json_result, markdown_result
    
    pipeline.add_preprocess_hook(sample_preprocess)
    pipeline.add_postprocess_hook(sample_postprocess)
    print("自定义钩子添加完成")
    
    print("\n" + "=" * 50)
    print("示例完成！")
    print("注意: 实际使用时，需要配置 glmocr 模型路径")
    print("然后调用 pipeline.process() 来处理文档")


if __name__ == "__main__":
    main()
