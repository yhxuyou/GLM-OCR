#!/usr/bin/env bash
# ============================================================
# vLLM GLM-OCR 生产级启动脚本
# ============================================================
# 使用方式:
#   ./vllm-serve.sh                          # 默认配置启动
#   ./vllm-serve.sh --quant awq              # AWQ 量化模式
#   ./vllm-serve.sh --tp 2                   # 2卡张量并行
#   ./vllm-serve.sh --model /path/to/local   # 本地模型路径
# ============================================================
set -euo pipefail

# ── 默认配置 ──────────────────────────────────────────────────────────
MODEL_PATH="${VLLM_MODEL_PATH:-THUDM/glm-ocr}"
MODEL_NAME="${VLLM_MODEL_NAME:-glm-ocr}"
PORT="${VLLM_PORT:-8000}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
PIPELINE_PARALLEL="${VLLM_PIPELINE_PARALLEL:-1}"
QUANTIZATION="${VLLM_QUANTIZATION:-}"
DTYPE="${VLLM_DTYPE:-auto}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCHEDULING="${VLLM_SCHEDULING:-async}"

# ── 颜色 ──────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }

# ── 参数解析 ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --name)  MODEL_NAME="$2"; shift 2 ;;
        --port)  PORT="$2"; shift 2 ;;
        --tp)    TENSOR_PARALLEL="$2"; shift 2 ;;
        --pp)    PIPELINE_PARALLEL="$2"; shift 2 ;;
        --quant) QUANTIZATION="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --gmem)  GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --cuda)  CUDA_DEVICES="$2"; shift 2 ;;
        --sched) SCHEDULING="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH     Model path or HF ID (default: THUDM/glm-ocr)"
            echo "  --name NAME      Served model name (default: glm-ocr)"
            echo "  --port PORT      Server port (default: 8000)"
            echo "  --tp N           Tensor parallelism (default: 1)"
            echo "  --pp N           Pipeline parallelism (default: 1)"
            echo "  --quant TYPE     Quantization: awq, gptq, fp8 (default: none)"
            echo "  --dtype TYPE     Model dtype: auto, half, float16, bfloat16 (default: auto)"
            echo "  --gmem FLOAT     GPU memory utilization 0-1 (default: 0.90)"
            echo "  --max-len INT    Max model length (default: 8192)"
            echo "  --max-seqs INT   Max concurrent sequences (default: 256)"
            echo "  --cuda DEVS      CUDA devices (default: 0)"
            echo "  --sched TYPE     Scheduling: async, sync (default: async)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# ── GPU 信息检测 ─────────────────────────────────────────────────────
log_info "检测 GPU 配置..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ' | sed 's/,$//')
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

    log_ok "检测到 $GPU_COUNT 张 GPU: $GPU_NAMES"
    log_info "单卡显存: $GPU_MEM"

    # 自动推荐 TP 配置
    if [ "$TENSOR_PARALLEL" -eq 1 ] && [ "$GPU_COUNT" -ge 2 ]; then
        log_warn "检测到 $GPU_COUNT 张 GPU，建议使用 --tp $GPU_COUNT 启用张量并行"
    fi
else
    log_warn "未检测到 NVIDIA GPU，vLLM 需要 GPU 才能运行"
    exit 1
fi

# ── 量化参数构建 ──────────────────────────────────────────────────────
QUANT_ARGS=""
if [ -n "$QUANTIZATION" ]; then
    case "$QUANTIZATION" in
        awq)
            QUANT_ARGS="--quantization awq"
            log_info "使用 AWQ 量化（推荐，精度损失最小）"
            ;;
        gptq)
            QUANT_ARGS="--quantization gptq"
            log_info "使用 GPTQ 量化"
            ;;
        fp8)
            QUANT_ARGS="--quantization fp8"
            log_info "使用 FP8 量化（需要 H100/H200）"
            ;;
        *)
            log_warn "未知量化类型: $QUANTIZATION，忽略"
            ;;
    esac
fi

# ── 调度策略 ──────────────────────────────────────────────────────────
SCHED_ARGS=""
case "$SCHEDULING" in
    async)
        SCHED_ARGS="--enable-chunked-prefill --num-scheduler-steps 8"
        log_info "使用异步调度 + Chunked Prefill"
        ;;
    sync)
        SCHED_ARGS=""  # vLLM 默认同步调度
        log_info "使用同步调度（默认）"
        ;;
esac

# ── 生成 vLLM 启动命令 ────────────────────────────────────────────────
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model \"$MODEL_PATH\" \
    --served-model-name \"$MODEL_NAME\" \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --pipeline-parallel-size $PIPELINE_PARALLEL \
    --dtype $DTYPE \
    --trust-remote-code \
    --enforce-eager \
    --max-num-batched-tokens $(($MAX_MODEL_LEN * 2)) \
    $QUANT_ARGS $SCHED_ARGS"

# ── 显示配置 ──────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  vLLM GLM-OCR 服务启动"
echo "============================================"
echo "  模型:       $MODEL_PATH"
echo "  服务名称:   $MODEL_NAME"
echo "  端口:       $PORT"
echo "  TP/PP:      $TENSOR_PARALLEL / $PIPELINE_PARALLEL"
echo "  量化:       ${QUANTIZATION:-无}"
echo "  显存利用:   $GPU_MEM_UTIL"
echo "  Max Length: $MAX_MODEL_LEN"
echo "  Max Seqs:   $MAX_NUM_SEQS"
echo "  CUDA:       $CUDA_DEVICES"
echo "  Dtype:      $DTYPE"
echo "  调度:       $SCHEDULING"
echo "============================================"
echo ""

# ── 启动服务 ──────────────────────────────────────────────────────────
log_info "启动 vLLM 服务..."
log_info "命令: $VLLM_CMD"
echo ""

eval "$VLLM_CMD"