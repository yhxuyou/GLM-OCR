#!/bin/bash
# Medical OCR Server Startup Script
# Usage: ./start_server.sh [dev|prod|docker] [--high-perf]

set -e

# Configuration
APP_MODULE="medical_ocr.high_perf_server:app"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT=8080
DEFAULT_WORKERS=4
LOG_DIR="./logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Parse arguments
ENV=${1:-prod}
HIGH_PERF=false

for arg in "$@"; do
    case $arg in
        --high-perf)
            HIGH_PERF=true
            DEFAULT_WORKERS=8
            ;;
    esac
done

case $ENV in
    dev)
        echo "Starting Medical OCR in DEVELOPMENT mode"

        if [ "$HIGH_PERF" = true ]; then
            echo "Using high-performance mode (single worker)"
            python -m "$APP_MODULE" \
                --host "$DEFAULT_HOST" \
                --port "$DEFAULT_PORT" \
                --workers 2 \
                --log-level DEBUG
        else
            uvicorn "$APP_MODULE" \
                --host "$DEFAULT_HOST" \
                --port "$DEFAULT_PORT" \
                --reload \
                --log-level info
        fi
        ;;

    prod)
        echo "Starting Medical OCR in PRODUCTION mode"

        pip install gunicorn

        gunicorn "$APP_MODULE" \
            --bind "$DEFAULT_HOST":"$DEFAULT_PORT" \
            --workers "$DEFAULT_WORKERS" \
            --threads 4 \
            --timeout 120 \
            --worker-class sync \
            --access-logfile "$LOG_DIR"/access.log \
            --error-logfile "$LOG_DIR"/error.log \
            --capture-output \
            --daemon \
            --pid "$LOG_DIR"/server.pid

        echo "Server started with PID $(cat "$LOG_DIR"/server.pid)"
        ;;

    docker)
        echo "Starting Medical OCR with Docker Compose"

        if [ ! -f .env ]; then
            echo "Creating default .env file..."
            cat > .env << EOF
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_ENABLED=true
THREAD_POOL_SIZE=8
PROMETHEUS_ENABLED=false
EOF
        fi

        docker-compose up -d

        echo "Docker containers started"
        docker-compose ps
        ;;

    *)
        echo "Usage: $0 [dev|prod|docker] [--high-perf]"
        echo ""
        echo "Options:"
        echo "  dev    - Development mode (hot reload)"
        echo "  prod   - Production mode (gunicorn, multiple workers)"
        echo "  docker - Docker Compose deployment"
        echo "  --high-perf - Use high-performance settings (more workers)"
        exit 1
        ;;
esac