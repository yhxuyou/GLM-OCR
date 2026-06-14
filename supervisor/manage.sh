#!/bin/bash
# 服务管理脚本 - 使用 Supervisor 管理微服务

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONF_DIR="$SCRIPT_DIR/conf.d"
LOG_DIR="/var/log/supervisor"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Supervisor 是否运行
check_supervisor() {
    if ! supervisorctl status >/dev/null 2>&1; then
        log_error "Supervisor 未运行"
        exit 1
    fi
}

# 生成配置文件
generate_configs() {
    local preprocess_procs=${1:-2}
    local glmocr_procs=${2:-4}
    local postprocess_procs=${3:-2}

    log_info "生成配置文件..."
    log_info "  预处理服务: $preprocess_procs 个进程"
    log_info "  GLM OCR 服务: $glmocr_procs 个进程"
    log_info "  后处理服务: $postprocess_procs 个进程"

    python3 "$SCRIPT_DIR/generate_configs.py" \
        --preprocess-procs "$preprocess_procs" \
        --glmocr-procs "$glmocr_procs" \
        --postprocess-procs "$postprocess_procs" \
        --workspace "$PROJECT_DIR" \
        --log-dir "$LOG_DIR" \
        --output-dir "$CONF_DIR"

    log_info "配置文件生成完成"
}

# 启动所有服务
start_all() {
    check_supervisor
    log_info "启动所有服务..."
    supervisorctl reread
    supervisorctl update
    supervisorctl start all
    log_info "所有服务已启动"
}

# 停止所有服务
stop_all() {
    check_supervisor
    log_info "停止所有服务..."
    supervisorctl stop all
    log_info "所有服务已停止"
}

# 重启所有服务
restart_all() {
    check_supervisor
    log_info "重启所有服务..."
    supervisorctl restart all
    log_info "所有服务已重启"
}

# 启动指定服务
start_service() {
    local service=$1
    check_supervisor
    log_info "启动服务: $service"
    supervisorctl start "${service}:*"
}

# 停止指定服务
stop_service() {
    local service=$1
    check_supervisor
    log_info "停止服务: $service"
    supervisorctl stop "${service}:*"
}

# 重启指定服务
restart_service() {
    local service=$1
    check_supervisor
    log_info "重启服务: $service"
    supervisorctl restart "${service}:*"
}

# 查看服务状态
status() {
    check_supervisor
    supervisorctl status
}

# 查看服务日志
logs() {
    local service=$1
    local lines=${2:-50}
    
    if [ -z "$service" ]; then
        log_error "请指定服务名称: preprocess, glmocr-async, postprocess"
        exit 1
    fi

    local log_file="$LOG_DIR/${service}-00.log"
    if [ ! -f "$log_file" ]; then
        log_error "日志文件不存在: $log_file"
        exit 1
    fi

    tail -n "$lines" -f "$log_file"
}

# 扩展/收缩服务进程数
scale() {
    local service=$1
    local num_procs=$2

    if [ -z "$service" ] || [ -z "$num_procs" ]; then
        log_error "用法: $0 scale <service> <num_procs>"
        log_error "示例: $0 scale preprocess 4"
        exit 1
    fi

    check_supervisor

    log_info "调整服务 $service 进程数为 $num_procs"
    
    # 重新生成配置
    local conf_file="$CONF_DIR/${service}.conf"
    if [ ! -f "$conf_file" ]; then
        log_error "配置文件不存在: $conf_file"
        exit 1
    fi

    # 获取当前配置
    local current_preprocess=$(grep -c "^\[program:preprocess-" "$CONF_DIR/preprocess.conf" 2>/dev/null || echo "2")
    local current_glmocr=$(grep -c "^\[program:glmocr-async-" "$CONF_DIR/glmocr-async.conf" 2>/dev/null || echo "4")
    local current_postprocess=$(grep -c "^\[program:postprocess-" "$CONF_DIR/postprocess.conf" 2>/dev/null || echo "2")

    # 更新对应服务的进程数
    case $service in
        preprocess)
            current_preprocess=$num_procs
            ;;
        glmocr-async)
            current_glmocr=$num_procs
            ;;
        postprocess)
            current_postprocess=$num_procs
            ;;
        *)
            log_error "未知服务: $service"
            exit 1
            ;;
    esac

    # 重新生成配置
    generate_configs "$current_preprocess" "$current_glmocr" "$current_postprocess"

    # 重新加载配置
    supervisorctl reread
    supervisorctl update
    
    log_info "服务 $service 已调整为 $num_procs 个进程"
}

# 显示帮助信息
help() {
    cat << EOF
用法: $0 <command> [options]

命令:
  generate [preprocess_procs] [glmocr_procs] [postprocess_procs]
                           生成 Supervisor 配置文件
  start                    启动所有服务
  stop                     停止所有服务
  restart                  重启所有服务
  start <service>          启动指定服务
  stop <service>           停止指定服务
  restart <service>        重启指定服务
  status                   查看所有服务状态
  logs <service> [lines]   查看服务日志
  scale <service> <num>    调整服务进程数
  help                     显示此帮助信息

服务名称:
  preprocess               预处理服务
  glmocr-async             GLM OCR 异步服务
  postprocess              后处理服务

示例:
  $0 generate 2 4 2        生成配置：2个预处理、4个OCR、2个后处理
  $0 start                 启动所有服务
  $0 status                查看服务状态
  $0 scale preprocess 4    将预处理服务扩展到4个进程
  $0 logs glmocr-async 100 查看 OCR 服务最近100行日志
EOF
}

# 主命令处理
case "${1:-}" in
    generate)
        shift
        generate_configs "$@"
        ;;
    start)
        shift
        if [ -n "${1:-}" ]; then
            start_service "$1"
        else
            start_all
        fi
        ;;
    stop)
        shift
        if [ -n "${1:-}" ]; then
            stop_service "$1"
        else
            stop_all
        fi
        ;;
    restart)
        shift
        if [ -n "${1:-}" ]; then
            restart_service "$1"
        else
            restart_all
        fi
        ;;
    status)
        status
        ;;
    logs)
        shift
        logs "$@"
        ;;
    scale)
        shift
        scale "$@"
        ;;
    help|--help|-h)
        help
        ;;
    *)
        log_error "未知命令: ${1:-}"
        help
        exit 1
        ;;
esac
