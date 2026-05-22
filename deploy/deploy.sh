#!/usr/bin/env bash
# ============================================================
# GLM-OCR Production Deployment Script
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.production"

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ── Pre-flight Checks ───────────────────────────────────────────────
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Install it first: https://docs.docker.com/engine/install/"
        exit 1
    fi
    log_ok "Docker found: $(docker --version)"

    if ! command -v docker compose &> /dev/null; then
        log_error "Docker Compose is not installed. Install it first."
        exit 1
    fi
    log_ok "Docker Compose found: $(docker compose version)"

    if nvidia-smi &> /dev/null; then
        log_ok "NVIDIA GPU detected"
        HAS_GPU=true
    else
        log_warn "No NVIDIA GPU detected. GPU-accelerated services will not be available."
        log_warn "Use --cpu-only flag to deploy without GPU services."
        HAS_GPU=false
    fi
}

# ── Environment Setup ───────────────────────────────────────────────
setup_env() {
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating .env.production from template..."
        cat > "$ENV_FILE" << 'ENVEOF'
# GLM-OCR Production Environment
# Copy this file and fill in your values

# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# vLLM Model Path (local path or HuggingFace model ID)
VLLM_MODEL_PATH=THUDM/glm-ocr

# HuggingFace token (required for gated models)
HF_TOKEN=

# RabbitMQ
RABBITMQ_PASSWORD=glmocr_pass

# Grafana
GRAFANA_PASSWORD=admin

# API Keys (if using MaaS mode)
ZHIPU_API_KEY=
ENVEOF
        log_ok "Created $ENV_FILE - please fill in your values"
    else
        log_ok "Environment file found: $ENV_FILE"
    fi

    set -a
    source "$ENV_FILE"
    set +a
}

# ── Docker Compose Operations ───────────────────────────────────────
deploy_docker() {
    local profile="default"
    if [ "$HAS_GPU" = true ]; then
        profile="full"
    fi

    log_info "Deploying GLM-OCR with profile: $profile"
    cd "$SCRIPT_DIR/docker-compose"

    docker compose \
        --env-file "$ENV_FILE" \
        -f docker-compose.yml \
        --profile "$profile" \
        build --pull

    docker compose \
        --env-file "$ENV_FILE" \
        -f docker-compose.yml \
        --profile "$profile" \
        up -d

    log_info "Waiting for services to be healthy..."
    sleep 10
    docker compose \
        --env-file "$ENV_FILE" \
        -f docker-compose.yml \
        --profile "$profile" \
        ps

    log_ok "Deployment complete!"
    log_info "Services:"
    echo "  Backend API:     http://localhost:8001"
    echo "  Pipeline API:    http://localhost:5002"
    echo "  RabbitMQ Admin:  http://localhost:15672"
    echo "  Prometheus:      http://localhost:9090"
    echo "  Grafana:         http://localhost:3000"
    if [ "$HAS_GPU" = true ]; then
        echo "  vLLM Server:     http://localhost:8000"
    fi
}

deploy_docker_cpu() {
    log_info "Deploying GLM-OCR in CPU-only mode..."
    log_warn "Only the backend and pipeline (without GPU) will be deployed."
    log_warn "The VLM model must be accessed via MaaS mode or external service."

    cd "$SCRIPT_DIR/docker-compose"
    docker compose \
        --env-file "$ENV_FILE" \
        -f docker-compose.yml \
        up -d redis rabbitmq glmocr-backend prometheus grafana
}

# ── Kubernetes Operations ──────────────────────────────────────────
deploy_k8s() {
    log_info "Deploying GLM-OCR to Kubernetes..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed."
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi

    cd "$SCRIPT_DIR/k8s"

    # Create namespace first
    kubectl apply -f namespace.yaml

    # Create PVC
    kubectl apply -f pvc.yaml

    # Create ConfigMap
    kubectl apply -f configmap.yaml

    # Deploy stateless services
    kubectl apply -f backend-deployment.yaml

    # Deploy GPU service (if GPU nodes available)
    if kubectl get nodes -l nvidia.com/gpu.present=true &> /dev/null 2>&1; then
        kubectl apply -f vllm-deployment.yaml
        kubectl apply -f pipeline-deployment.yaml
    else
        log_warn "No GPU nodes detected. Skipping vLLM and Pipeline deployments."
        log_warn "Configure the backend to use MaaS mode or external OCR service."
    fi

    # Create HPA
    kubectl apply -f hpa.yaml

    # Wait for deployments
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s \
        -n glm-ocr deployment/glmocr-backend || true

    log_ok "Kubernetes deployment complete!"
}

# ── Monitoring Setup ────────────────────────────────────────────────
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    cd "$SCRIPT_DIR/docker-compose"

    docker compose \
        --env-file "$ENV_FILE" \
        -f docker-compose.yml \
        up -d prometheus grafana

    log_ok "Monitoring stack deployed!"
}

# ── Health Check ─────────────────────────────────────────────────────
check_health() {
    log_info "Running health checks..."

    local endpoints=(
        "http://localhost:8001/health:Backend"
        "http://localhost:5002/health:Pipeline"
    )

    for endpoint in "${endpoints[@]}"; do
        local url="${endpoint%%:*}"
        local name="${endpoint##*:}"
        if curl -sf "$url" > /dev/null 2>&1; then
            log_ok "$name is healthy ($url)"
        else
            log_warn "$name is not responding ($url)"
        fi
    done
}

# ── Cleanup ──────────────────────────────────────────────────────────
cleanup() {
    log_warn "This will remove all GLM-OCR containers and volumes!"
    read -p "Are you sure? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$SCRIPT_DIR/docker-compose"
        docker compose -f docker-compose.yml down -v
        log_ok "Cleanup complete"
    fi
}

# ── Main ────────────────────────────────────────────────────────────
main() {
    local cmd="${1:-help}"

    case "$cmd" in
        check)
            check_prerequisites
            ;;
        setup)
            check_prerequisites
            setup_env
            ;;
        docker)
            check_prerequisites
            setup_env
            deploy_docker
            check_health
            ;;
        docker:cpu)
            check_prerequisites
            setup_env
            deploy_docker_cpu
            ;;
        k8s)
            deploy_k8s
            ;;
        monitoring)
            deploy_monitoring
            ;;
        health)
            check_health
            ;;
        clean)
            cleanup
            ;;
        help|*)
            echo "GLM-OCR Production Deployment Script"
            echo ""
            echo "Usage: $0 <command>"
            echo ""
            echo "Commands:"
            echo "  check          Check prerequisites only"
            echo "  setup          Create .env.production file"
            echo "  docker         Full Docker Compose deployment (GPU)"
            echo "  docker:cpu     CPU-only Docker Compose deployment"
            echo "  k8s            Deploy to Kubernetes"
            echo "  monitoring     Deploy monitoring stack only"
            echo "  health         Run health checks"
            echo "  clean          Remove all containers and volumes"
            echo ""
            echo "Examples:"
            echo "  # First time setup"
            echo "  $0 setup"
            echo "  # Edit .env.production then deploy"
            echo "  $0 docker"
            echo ""
            echo "For Kubernetes:"
            echo "  # Build and push Docker images first:"
            echo "  docker build -f deploy/docker/glmocr-backend.Dockerfile -t your-registry/glmocr-backend:latest ."
            echo "  docker build -f deploy/docker/glmocr-pipeline.Dockerfile -t your-registry/glmocr-pipeline:latest ."
            echo "  docker push your-registry/glmocr-backend:latest"
            echo "  docker push your-registry/glmocr-pipeline:latest"
            echo "  # Then edit k8s/*.yaml to use your registry, then:"
            echo "  $0 k8s"
            ;;
    esac
}

main "$@"