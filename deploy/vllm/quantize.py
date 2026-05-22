#!/usr/bin/env python3
"""
GLM-OCR 模型量化工具

将 GLM-OCR 模型量化为 AWQ/GPTQ 格式，减少显存占用 60-80%，
同时保持高精度。

使用方式:
    python quantize.py --model THUDM/glm-ocr --output ./quantized/glm-ocr-awq --method awq
"""
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def check_dependencies(method: str):
    """Check that the required quantization library is installed."""
    if method == "awq":
        try:
            import auto_gptq  # noqa: F401
        except ImportError:
            logger.error(
                "AWQ quantization requires auto_gptq. "
                "Install: pip install auto-gptq"
            )
            raise
    elif method == "gptq":
        try:
            import auto_gptq  # noqa: F401
        except ImportError:
            logger.error(
                "GPTQ quantization requires auto_gptq. "
                "Install: pip install auto-gptq"
            )
            raise


def quantize_awq(
    model_path: str,
    output_path: str,
    quantize_config: str = "int4",
    calib_size: int = 128,
):
    """Quantize model using AWQ (Activation-aware Weight Quantization).

    Args:
        model_path: Path to the original model.
        output_path: Path to save the quantized model.
        quantize_config: Quantization bits config (int4, int8, etc).
        calib_size: Number of calibration samples.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    logger.info(f"Loading model from {model_path}...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    logger.info(f"Quantizing with AWQ ({quantize_config})...")
    model.quantize(
        tokenizer=tokenizer,
        quant_config=quantize_config,
        calib_size=calib_size,
    )

    logger.info(f"Saving quantized model to {output_path}...")
    model.save_quantized(output_path, safetensors=True)
    tokenizer.save_pretrained(output_path)

    logger.info("Quantization complete!")
    logger.info(f"Original model: {model_path}")
    logger.info(f"Quantized model: {output_path}")
    logger.info(f"Quantization: AWQ {quantize_config}")

    # 输出显存节省估算
    logger.info("")
    logger.info("Estimated memory savings (compared to FP16):")
    logger.info("  FP16: ~1.8 GB (for 0.9B params)")
    if "int4" in quantize_config:
        logger.info("  INT4: ~0.5 GB (75% reduction)")
    elif "int8" in quantize_config:
        logger.info("  INT8: ~0.9 GB (50% reduction)")


def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    calib_size: int = 128,
):
    """Quantize model using GPTQ.

    Args:
        model_path: Path to the original model.
        output_path: Path to save the quantized model.
        bits: Number of bits (4 or 8).
        group_size: Group size for quantization (128 is recommended).
        calib_size: Number of calibration samples.
    """
    from auto_gptq import AutoGPTQForCausalLM
    from transformers import AutoTokenizer

    logger.info(f"Loading model from {model_path}...")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    logger.info(f"Quantizing with GPTQ (bits={bits}, group_size={group_size})...")
    model.quantize(
        tokenizer=tokenizer,
        bits=bits,
        group_size=group_size,
        calib_size=calib_size,
    )

    logger.info(f"Saving quantized model to {output_path}...")
    model.save_quantized(output_path, safetensors=True)
    tokenizer.save_pretrained(output_path)

    logger.info("Quantization complete!")
    logger.info(f"Original model: {model_path}")
    logger.info(f"Quantized model: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize GLM-OCR model for production deployment"
    )
    parser.add_argument(
        "--model",
        default="THUDM/glm-ocr",
        help="Model path or HuggingFace ID (default: THUDM/glm-ocr)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the quantized model",
    )
    parser.add_argument(
        "--method",
        choices=["awq", "gptq"],
        default="awq",
        help="Quantization method (default: awq, recommended for GLM-OCR)",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=128,
        help="Calibration dataset size (default: 128)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits for GPTQ (default: 4)",
    )
    args = parser.parse_args()

    check_dependencies(args.method)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.method == "awq":
        quantize_awq(
            model_path=args.model,
            output_path=str(output_path),
            quantize_config=f"int{args.bits}_group_size_128",
            calib_size=args.calib_size,
        )
    elif args.method == "gptq":
        quantize_gptq(
            model_path=args.model,
            output_path=str(output_path),
            bits=args.bits,
            group_size=128,
            calib_size=args.calib_size,
        )


if __name__ == "__main__":
    main()