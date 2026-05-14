#!/bin/bash

# GLM-OCR FastAPI服务启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "======================================"
echo "  GLM-OCR FastAPI Service"
echo "======================================"
echo ""

# 检查Python版本
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v $cmd >/dev/null 2>&1; then
        PYTHON_VERSION=$($cmd --version | awk '{print $2}')
        if [[ "$PYTHON_VERSION" =~ ^3\.(8|9|10|11) ]]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.8+ not found!"
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"
echo ""

# 检查是否安装依赖
cd "$PROJECT_ROOT"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "Installing dependencies..."
pip install -q -e ".[all]"
pip install -q -r "deploy/fastapi/requirements.txt"

echo ""
echo "Starting GLM-OCR FastAPI service..."
echo "Service will be available at: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the service"
echo "======================================"
echo ""

# 启动服务
uvicorn deploy.fastapi.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
