"""
医疗费用清单完整 OCR 工作流示例

演示如何将 MedicalExpenseLayoutDetector 与完整的 GLM-OCR 工作流结合使用。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
from glmocr.config import load_config
from glmocr.pipeline import Pipeline
from glmocr.layout.medical_detector import create_medical_expense_detector
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class MedicalExpensePipeline:
    """医疗费用清单完整处理管道"""

    def __init__(self, config_path=None):
        """初始化"""
        self.config = load_config(config_path)
        self.pipeline = None
        self.medical_detector = None
        self._initialized = False

    def initialize(self):
        """初始化管道"""
        logger.info("初始化医疗费用清单处理管道...")

        # 创建医疗费用清单布局检测器
        logger.info("创建医疗费用清单布局检测器...")
        self.medical_detector = create_medical_expense_detector(
            self.config.layout,
            table_expand_ratio=0.05,
            table_min_expansion=10,
        )

        # 启动医疗费用清单布局检测器
        logger.info("启动医疗费用清单布局检测器...")
        self.medical_detector.start()

        # 创建并启动完整 OCR 管道（使用标准布局检测器）
        # 注意: 这里是为了演示，实际中你可以选择使用医疗检测器或者标准检测器
        logger.info("创建完整 OCR 管道...")
        self.pipeline = Pipeline(config=self.config.pipeline)
        self.pipeline.start()

        self._initialized = True
        logger.info("医疗费用清单处理管道初始化完成！")

    def shutdown(self):
        """关闭管道"""
        logger.info("关闭医疗费用清单处理管道...")

        if self.medical_detector:
            self.medical_detector.stop()
            self.medical_detector = None

        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

        self._initialized = False
        logger.info("医疗费用清单处理管道已关闭。")

    def process_image(self, image_path, output_dir="./medical_output"):
        """处理单张医疗费用清单图像

        执行流程:
        1. 使用医疗费用清单检测器分割表格和非表格区域
        2. 对表格区域使用 OCR 识别
        3. 对非表格整体图使用 OCR 识别
        4. 保存所有结果
        """
        if not self._initialized:
            raise RuntimeError("管道未初始化，请先调用 initialize()")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"处理图像: {image_path}")

        # 1. 加载图像
        image = Image.open(image_path)

        # 2. 使用医疗费用清单检测器分割区域
        logger.info("运行医疗费用清单布局检测...")
        _, _, medical_results = self.medical_detector.process(
            [image],
            save_visualization=True,
            return_medical_result=True,
        )

        medical_result = medical_results[0]

        # 3. 保存医疗费用清单分割结果
        logger.info("保存区域分割结果...")
        saved_files = medical_result.save(str(output_path))

        # 4. 对表格区域进行 OCR 识别
        ocr_results = {"tables": [], "non_table": None}

        for table in medical_result.table_regions:
            if table.cropped_image:
                logger.info(f"识别表格 {table.table_id}...")
                # 对表格裁剪图进行 OCR
                table_ocr_result = self._ocr_image(table.cropped_image)
                ocr_results["tables"].append({
                    "table_id": table.table_id,
                    "bbox": table.bbox_pixel,
                    "ocr": table_ocr_result,
                })

        # 5. 对非表格整体图进行 OCR 识别
        if medical_result.non_table_image:
            logger.info("识别非表格整体图...")
            non_table_ocr = self._ocr_image(medical_result.non_table_image.merged_image)
            ocr_results["non_table"] = non_table_ocr

        # 6. 保存 OCR 结果
        self._save_ocr_results(ocr_results, output_path)

        logger.info(f"处理完成！结果已保存到: {output_dir}")
        return {
            "medical_result": medical_result,
            "ocr_results": ocr_results,
            "saved_files": saved_files,
        }

    def _ocr_image(self, image):
        """对单张图像进行 OCR 识别"""
        if not self.pipeline:
            raise RuntimeError("OCR 管道未初始化")

        # 使用完整管道识别
        result_generator = self.pipeline.parse(
            image=image,
            save_layout_visualization=False,
        )

        # 获取第一个结果
        for result in result_generator:
            return {
                "json": result.json_result,
                "markdown": result.markdown_result,
                "raw": result.raw_json_result,
            }
        return None

    def _save_ocr_results(self, ocr_results, output_path):
        """保存 OCR 结果"""
        import json

        # 保存 JSON 格式
        with open(output_path / "ocr_results.json", "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)

        # 保存 Markdown 格式（合并结果）
        with open(output_path / "ocr_results.md", "w", encoding="utf-8") as f:
            f.write("# 医疗费用清单 OCR 结果\n\n")

            if ocr_results["non_table"]:
                f.write("## 非表格内容\n\n")
                f.write(ocr_results["non_table"]["markdown"])
                f.write("\n\n")

            f.write("## 表格内容\n\n")
            for table in ocr_results["tables"]:
                f.write(f"### 表格 {table['table_id']}\n\n")
                f.write(table["ocr"]["markdown"])
                f.write("\n\n")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="医疗费用清单完整 OCR 处理示例"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="配置文件路径"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="输入图像路径"
    )
    parser.add_argument(
        "--output", type=str, default="./medical_output", help="输出目录"
    )
    args = parser.parse_args()

    # 使用上下文管理器（自动处理初始化和关闭）
    with MedicalExpensePipeline(args.config) as pipeline:
        result = pipeline.process_image(args.image, args.output)

        print(f"\n处理结果:")
        print(f"  表格数量: {len(result['medical_result'].table_regions)}")
        print(f"  结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
