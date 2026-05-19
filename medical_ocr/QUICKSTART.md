# 快速开始指南

## 方法一：不安装直接运行（推荐用于测试）

### 1. 确保 glmocr 已安装

```bash
cd /workspace
pip install -e .  # 安装原始的 glmocr 项目
```

### 2. 安装额外依赖

```bash
cd /workspace/medical_ocr
pip install flask opencv-python numpy
```

### 3. 启动服务器

```bash
# 方法 A：直接运行 server.py（需要先设置 Python 路径）
cd /workspace/medical_ocr
python -m medical_ocr.server

# 方法 B：或者稍后我们可以创建一个简单的启动脚本
```

### 4. 测试 API

健康检查：
```bash
curl http://localhost:8080/health
```

## 项目文件说明

| 文件 | 说明 |
|------|------|
| `medical_ocr/pipeline.py` | 继承自 glmocr.Pipeline，添加预处理/后处理 |
| `medical_ocr/layout_detector.py` | 继承自 glmocr.PPDocLayoutDetector，优化医疗文档检测 |
| `medical_ocr/server.py` | Flask 服务器，提供 REST API |
| `examples/basic_usage.py` | 基础使用示例 |
| `run_tests.py` | 项目结构测试脚本（无需安装） |

## 当前状态

✅ 项目结构完整
✅ 代码已创建，继承关系正确
✅ 可以在安装 glmocr 后直接使用

## 下一步

需要我帮您：
1. 创建一个一键启动脚本？
2. 准备一个测试用的配置文件？
3. 创建一个简单的测试客户端来调用 API？

