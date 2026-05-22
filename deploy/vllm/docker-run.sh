#!/usr/bin/env bash
# ============================================================
# GLM-OCR vLLM Docker 启动方案
# ============================================================
# 提供三种模式:
#   1. 标准模式 - 直接从 HuggingFace 加载模型
#   2. 量化模式 - 使用 AWQ/GPTQ 量化模型 (更省显存)
#   3. 本地模式 - 从本地路径加载已下载的模型
# ============================================================
set -euo pipefail

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }

# ── 前置检查 ─────────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    log_error "Docker 未安装"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    log_error "未检测到 NVIDIA GPU"
    exit 1
fi

# ── 配置 ─────────────────────────────────────────────────────────────
MODEL_PATH="${1:-THUDM/glm-ocr}"
HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
CACHE_DIR="${VLLM_CACHE_DIR:-./model-cache}"

# 模型名称 (用于 --served-model-name)
MODEL_NAME="${2:-glm-ocr}"

# 量化类型: awq, gptq, fp8, 或留空使用 FP16
QUANT="${3:-}"

# 创建缓存目录
mkdir -p "$CACHE_DIR"

log_info "GLM-OCR vLLM Docker 启动"
log_info "  Model:    $MODEL_PATH"
log_info "  Name:     $MODEL_NAME"
log_info "  Quant:    ${QUANT:-FP16}"
log_info "  Cache:    $CACHE_DIR"

# ── 构建 Docker 参数 ─────────────────────────────────────────────────
DOCKER_ARGS=(
    --rm
    --gpus all
    --name glmocr-vllm
    -p 8000:8000
    -v "$CACHE_DIR:/root/.cache/huggingface"
)

# 量化参数
VLLM_ARGS=(
    --model "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --host 0.0.0.0
    --port 8000
    --gpu-memory-utilization 0.90
    --max-model-len 8192
    --max-num-seqs 256
    --trust-remote-code
    --enforce-eager
    --enable-chunked-prefill
    --num-scheduler-steps 8
)

if [ -n "$QUANT" ]; then
    case "$QUANT" in
        awq)
            VLLM_ARGS+=(--quantization awq)
            log_info "量化模式: AWQ INT4"
            ;;
        gptq)
            VLLM_ARGS+=(--quantization gptq)
            log_info "量化模式: GPTQ INT4"
            ;;
        fp8)
            VLLM_ARGS+=(--quantization fp8)
            log_info "量化模式: FP8"
            ;;
        *)
            log_warn "未知量化类型: $QUANT, 使用 FP16"
            ;;
    esac
fi

# HuggingFace Token (用于 gated model)
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    DOCKER_ARGS+=(-e "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN")
    log_info "使用 HF Token 认证"
fi

# ── 启动 Docker 容器 ─────────────────────────────────────────────────
echo ""
log_info "启动 vLLM Docker 容器..."
echo ""

docker run "${DOCKER_ARGS[@]}" \
    vllm/vllm-openai:latest \
    "${VLLM_ARGS[@]}"