#!/bin/bash
# Medical OCR Server Startup Script
# Usage: ./start_server.sh [dev|prod]

set -e

# Configuration
APP_NAME="medical-ocr"
APP_MODULE="medical_ocr.high_perf_server:app"
DEFAULT_PORT=8080
DEFAULT_WORKERS=4
LOG_DIR="./logs"

# Create log directory
mkdir -p $LOG_DIR

# Parse arguments
ENV=${1:-prod}

case $ENV in
    dev)
        echo "🚀 Starting $APP_NAME in DEVELOPMENT mode"
        pip install fastapi uvicorn redis prometheus-client pydantic
        
        # Run with hot reload
        uvicorn $APP_MODULE \
            --host 0.0.0.0 \
            --port $DEFAULT_PORT \
            --reload \
            --log-level info
        ;;
    
    prod)
        echo "🚀 Starting $APP_NAME in PRODUCTION mode"
        
        # Install production dependencies
        pip install fastapi uvicorn redis prometheus-client pydantic uvloop httptools
        
        # Run with multiple workers
        uvicorn $APP_MODULE \
            --host 0.0.0.0 \
            --port $DEFAULT_PORT \
            --workers $DEFAULT_WORKERS \
            --loop uvloop \
            --http httptools \
            --timeout-keep-alive 120 \
            --log-level info \
            --access-log \
            --error-log $LOG_DIR/error.log \
            > $LOG_DIR/access.log 2>&1 &
        
        # Save PID
        echo $! > $LOG_DIR/server.pid
        echo "✅ Server started with PID $(cat $LOG_DIR/server.pid)"
        echo "📊 Metrics available at http://localhost:8001/metrics"
        ;;
    
    *)
        echo "❌ Invalid environment: $ENV"
        echo "Usage: $0 [dev|prod]"
        exit 1
        ;;
esac
