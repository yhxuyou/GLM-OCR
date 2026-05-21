#!/bin/bash
# Medical OCR High-Performance Server Startup Script
# Usage: ./start_high_perf.sh [dev|prod]

set -e

# Configuration
APP_MODULE="medical_ocr.high_perf_server:app"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT=8080
DEFAULT_WORKERS=8
LOG_DIR="./logs"

# Create log directory
mkdir -p $LOG_DIR

# Parse arguments
ENV=${1:-prod}

case $ENV in
    dev)
        echo "🚀 Starting High-Performance Medical OCR in DEVELOPMENT mode"
        
        # Run development server
        python -m $APP_MODULE \
            --host $DEFAULT_HOST \
            --port $DEFAULT_PORT \
            --workers 2 \
            --log-level DEBUG
        ;;
    
    prod)
        echo "🚀 Starting High-Performance Medical OCR in PRODUCTION mode"
        
        # Production: Use gunicorn for better performance
        pip install gunicorn
        
        gunicorn $APP_MODULE \
            --bind $DEFAULT_HOST:$DEFAULT_PORT \
            --workers $DEFAULT_WORKERS \
            --threads 4 \
            --timeout 120 \
            --worker-class sync \
            --access-logfile $LOG_DIR/access.log \
            --error-logfile $LOG_DIR/error.log \
            --capture-output \
            --daemon \
            --pid $LOG_DIR/server.pid
        
        echo "✅ Server started with PID $(cat $LOG_DIR/server.pid)"
        ;;
    
    docker)
        echo "🚀 Starting High-Performance Medical OCR with Docker Compose"
        
        # Check if .env exists
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
        
        # Start with docker-compose
        docker-compose up -d
        
        echo "✅ Docker containers started"
        docker-compose ps
        ;;
    
    *)
        echo "❌ Invalid environment: $ENV"
        echo "Usage: $0 [dev|prod|docker]"
        echo ""
        echo "Options:"
        echo "  dev    - Development mode (single worker)"
        echo "  prod   - Production mode (gunicorn, multiple workers)"
        echo "  docker - Docker Compose deployment"
        exit 1
        ;;
esac
