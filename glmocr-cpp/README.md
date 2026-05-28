# GLM-OCR C++ 实现

这是 GLM-OCR 模型的 C++ 重实现版本，它采用与原始 Python 版本相同的架构，提供了更高效的性能。

## 特性

- 与 vLLM 部署的 GLM-OCR 模型兼容的 OpenAI API 接口
- 支持图像和 PDF 文档解析
- 多线程并行处理
- 可扩展的布局检测架构（当前为占位实现）
- JSON 和 Markdown 格式输出
- C++17 标准

## 架构

GLM-OCR C++ 实现包含以下核心模块：

```
glmocr-cpp/
├── include/glmocr/
│   ├── common.h           # 通用数据结构
│   ├── config.h           # 配置类
│   ├── page_loader.h      # 文档加载和图像预处理
│   ├── ocr_client.h       # OCR API 客户端
│   ├── result_formatter.h # 结果格式化
│   └── pipeline.h         # 主处理管道
├── src/
│   ├── config.cpp
│   ├── page_loader.cpp
│   ├── ocr_client.cpp
│   ├── result_formatter.cpp
│   ├── pipeline.cpp
│   └── utils/             # 工具函数
└── examples/
    ├── cli.cpp            # 命令行工具
    └── simple.cpp         # 简单示例
```

## 依赖

- C++17 兼容编译器
- CMake 3.15+
- libcurl (用于 HTTP 请求)
- nlohmann/json (自动获取)
- stb_image (自动获取)

## 构建

```bash
cd glmocr-cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 使用方法

### 1. 启动 vLLM 服务

首先需要用 vLLM 部署 GLM-OCR 模型：

```bash
vllm serve zai-org/GLM-OCR \
  --port 8080 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --served-model-name glm-ocr
```

### 2. 使用 C++ API

```cpp
#include "glmocr.h"

int main() {
    // 配置
    glmocr::GlmOcrConfig config;
    config.pipeline.ocr_api.api_host = "localhost";
    config.pipeline.ocr_api.api_port = 8080;
    
    // 创建 OCR 实例
    glmocr::GlmOcr ocr(config);
    
    // 解析文件
    auto result = ocr.parse("document.png");
    
    // 输出结果
    std::cout << result.markdown_output << std::endl;
    std::cout << result.json_result << std::endl;
    
    return 0;
}
```

### 3. 使用命令行工具

```bash
# 处理单个文件
./glmocr_cli document.png

# 指定 API 端点
./glmocr_cli --host localhost --port 8080 document.pdf

# 输出到指定目录
./glmocr_cli --output ./results *.png

# 仅输出 JSON
./glmocr_cli --json document.png
```

## 模块说明

### PageLoader
负责加载图像和 PDF 文档，执行图像预处理，并构建 OCR API 请求。

### OCRClient
HTTP 客户端，与 vLLM 服务通信，支持 OpenAI 兼容 API 和 Ollama API。

### ResultFormatter
处理 OCR 输出，格式化文本、表格和公式，支持 Markdown 和 JSON 输出。

### Pipeline
协调各模块工作，实现两阶段处理流程（布局检测 → 并行 OCR）。

### Layout Detection (待实现)
布局检测模块的占位实现，可扩展为使用 PP-DocLayout-V3 模型。

## 扩展布局检测

要实现完整的两阶段处理流程（布局检测 + 并行 OCR），可以：

1. 集成 PP-DocLayout-V3 模型的推理引擎（如 ONNX Runtime）
2. 替换 `detect_layout()` 函数
3. 实现区域裁剪和并行 OCR 调用

请参考原始 Python 代码的实现细节。

## 性能优化建议

- 增加 `max_workers` 配置项以支持更高并发
- 使用连接池复用 HTTP 连接
- 实现多线程布局检测
- 添加批处理支持

## 许可证

Apache License 2.0
