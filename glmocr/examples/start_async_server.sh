#!/usr/bin/env bash
# ============================================================================
# GLM-OCR 异步服务器启动脚本
# ============================================================================
#
# 本脚本用于启动 GLM-OCR 异步 FastAPI 服务器，支持文档的异步处理与实时进度推送。
#
# 主要功能：
#   - POST   /parse/async          提交文档进行异步处理
#   - GET    /parse/status/{doc_id} 查询处理进度
#   - GET    /parse/result/{doc_id} 获取最终结果
#   - WS     /ws/{doc_id}          WebSocket 实时进度推送
#
# 依赖安装：
#   pip install 'glmocr[server]'
#
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 环境变量配置示例（可根据实际情况修改）
# ---------------------------------------------------------------------------

# 服务器绑定地址与端口（默认 0.0.0.0:5002）
export GLMOCR_SERVER_HOST="${GLMOCR_SERVER_HOST:-0.0.0.0}"
export GLMOCR_SERVER_PORT="${GLMOCR_SERVER_PORT:-5002}"

# 日志级别：DEBUG / INFO / WARNING / ERROR
export GLMOCR_LOG_LEVEL="${GLMOCR_LOG_LEVEL:-INFO}"

# 智谱 API Key（MaaS 模式必需）
# export ZHIPU_API_KEY="your-api-key-here"

# 运行模式：maas（云端）或 selfhosted（本地 GPU）
# export GLMOCR_MODE="maas"

# Redis 连接地址（异步聚合器使用，默认本地 Redis）
# export GLMOCR_REDIS_URL="redis://localhost:6379/0"

# 自定义配置文件路径（可选，优先级低于环境变量）
# export GLMOCR_CONFIG_PATH="/path/to/config.yaml"

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 默认值
HOST="${GLMOCR_SERVER_HOST}"
PORT="${GLMOCR_SERVER_PORT}"
LOG_LEVEL="${GLMOCR_LOG_LEVEL}"
CONFIG_PATH="${GLMOCR_CONFIG_PATH:-}"

usage() {
    cat <<EOF
用法: $0 [选项]

选项:
  --host HOST          绑定地址（默认: 0.0.0.0）
  --port PORT          绑定端口（默认: 5002）
  --log-level LEVEL    日志级别: DEBUG|INFO|WARNING|ERROR（默认: INFO）
  --config PATH        配置文件路径（可选）
  --help               显示此帮助信息

环境变量:
  GLMOCR_SERVER_HOST   绑定地址
  GLMOCR_SERVER_PORT   绑定端口
  GLMOCR_LOG_LEVEL     日志级别
  ZHIPU_API_KEY        智谱 API Key（MaaS 模式必需）
  GLMOCR_MODE          运行模式（maas / selfhosted）
  GLMOCR_REDIS_URL     Redis 连接地址

示例:
  # 使用默认配置启动
  $0

  # 指定端口和日志级别
  $0 --port 8080 --log-level DEBUG

  # 使用自定义配置文件
  $0 --config /path/to/config.yaml
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --log-level)
            LOG_LEVEL="$2"; shift 2 ;;
        --config)
            CONFIG_PATH="$2"; shift 2 ;;
        --help)
            usage ;;
        *)
            echo "未知参数: $1"
            usage ;;
    esac
done

# ---------------------------------------------------------------------------
# 启动前检查
# ---------------------------------------------------------------------------

echo "=============================================="
echo " GLM-OCR 异步服务器启动脚本"
echo "=============================================="
echo ""

# 检查 Python 是否可用
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查 glmocr 是否已安装
if ! python3 -c "import glmocr" 2>/dev/null; then
    echo "[错误] glmocr 未安装，请执行: pip install 'glmocr[server]'"
    exit 1
fi

# 检查 FastAPI 依赖
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "[错误] FastAPI 未安装，请执行: pip install 'glmocr[server]'"
    exit 1
fi

echo "[信息] Python 版本: $(python3 --version)"
echo "[信息] 绑定地址: ${HOST}:${PORT}"
echo "[信息] 日志级别: ${LOG_LEVEL}"
if [[ -n "${CONFIG_PATH}" ]]; then
    echo "[信息] 配置文件: ${CONFIG_PATH}"
fi
echo ""

# ---------------------------------------------------------------------------
# 构建启动命令
# ---------------------------------------------------------------------------

CMD=(python3 -m glmocr.async_server)
CMD+=(--host "$HOST")
CMD+=(--port "$PORT")
CMD+=(--log-level "$LOG_LEVEL")

if [[ -n "${CONFIG_PATH}" ]]; then
    CMD+=(--config "$CONFIG_PATH")
fi

# ---------------------------------------------------------------------------
# 启动服务器
# ---------------------------------------------------------------------------

echo "[信息] 启动命令: ${CMD[*]}"
echo "[信息] 服务器启动中..."
echo ""
echo "API 端点:"
echo "  POST   http://${HOST}:${PORT}/parse/async          - 提交文档"
echo "  GET    http://${HOST}:${PORT}/parse/status/{doc_id} - 查询进度"
echo "  GET    http://${HOST}:${PORT}/parse/result/{doc_id} - 获取结果"
echo "  WS     ws://${HOST}:${PORT}/ws/{doc_id}             - WebSocket 进度推送"
echo "  GET    http://${HOST}:${PORT}/health                - 健康检查"
echo ""
echo "按 Ctrl+C 停止服务器"
echo "----------------------------------------------"

exec "${CMD[@]}"
