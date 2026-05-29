#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="${SCRIPT_DIR}/third_party"

echo "=== GLM-OCR C++ Offline Dependency Preparation ==="
echo "Target directory: ${THIRD_PARTY}"
echo ""

mkdir -p "${THIRD_PARTY}"

# 1. nlohmann/json (header-only)
NLOHMANN_DIR="${THIRD_PARTY}/nlohmann_json"
if [ ! -d "${NLOHMANN_DIR}" ]; then
    echo "[1/5] Downloading nlohmann/json v3.11.2 ..."
    git clone --depth 1 --branch v3.11.2 https://github.com/nlohmann/json.git "${NLOHMANN_DIR}"
    echo "  Done."
else
    echo "[1/5] nlohmann/json already exists, skipping."
fi

# 2. stb (header-only)
STB_DIR="${THIRD_PARTY}/stb"
if [ ! -d "${STB_DIR}" ]; then
    echo "[2/5] Downloading stb ..."
    git clone --depth 1 https://github.com/nothings/stb.git "${STB_DIR}"
    echo "  Done."
else
    echo "[2/5] stb already exists, skipping."
fi

# 3. yaml-cpp
YAML_CPP_DIR="${THIRD_PARTY}/yaml-cpp"
if [ ! -d "${YAML_CPP_DIR}" ]; then
    echo "[3/5] Downloading yaml-cpp 0.7.0 ..."
    git clone --depth 1 --branch yaml-cpp-0.7.0 https://github.com/jbeder/yaml-cpp.git "${YAML_CPP_DIR}"
    echo "  Done."
else
    echo "[3/5] yaml-cpp already exists, skipping."
fi

# 4. cpp-httplib
HTTPLIB_FILE="${THIRD_PARTY}/httplib.h"
if [ ! -f "${HTTPLIB_FILE}" ]; then
    echo "[4/5] Downloading cpp-httplib ..."
    curl -sL -o "${HTTPLIB_FILE}" https://github.com/yhirose/cpp-httplib/releases/download/v0.18.3/httplib.h
    echo "  Done."
else
    echo "[4/5] cpp-httplib already exists, skipping."
fi

# 5. ONNX Runtime
ORT_DIR="${THIRD_PARTY}/onnxruntime"
if [ ! -d "${ORT_DIR}" ] || [ ! -f "${ORT_DIR}/include/onnxruntime_cxx_api.h" ]; then
    echo "[5/5] Downloading ONNX Runtime 1.21.0 (Linux x64 CPU) ..."
    ORT_TMP="/tmp/onnxruntime-linux-x64-1.21.0"
    ORT_TGZ="/tmp/onnxruntime-linux-x64-1.21.0.tgz"
    curl -sL -o "${ORT_TGZ}" https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz
    tar xzf "${ORT_TGZ}" -C /tmp
    rm -rf "${ORT_DIR}"
    mv "${ORT_TMP}" "${ORT_DIR}"
    rm -f "${ORT_TGZ}"
    echo "  Done."
else
    echo "[5/5] ONNX Runtime already exists, skipping."
fi

echo ""
echo "=== All dependencies prepared! ==="
echo ""
echo "Directory structure:"
echo "  ${THIRD_PARTY}/"
echo "    nlohmann_json/   (header-only JSON library)"
echo "    stb/             (header-only image library)"
echo "    yaml-cpp/        (YAML library, will be built with project)"
echo "    httplib.h        (header-only HTTP library)"
echo "    onnxruntime/     (ONNX Runtime SDK)"
echo ""
echo "You can now copy the entire glmocr-cpp/ directory to the intranet machine."
echo "System dependencies to pre-install on the intranet machine:"
echo "  - libcurl-dev"
echo "  - cmake >= 3.15"
echo "  - g++ >= 9 (C++17 support)"
