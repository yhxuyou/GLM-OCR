# Single GPU - Multiple Pipelines Guide

## 🔍 原理说明

### Yes！可以在**单卡上运行多个 Pipeline**，但需要注意：

```
单 GPU，多个 Pipeline（线程池）

GPU 0
├─ Pipeline 1 ← Thread 1 处理请求 A
├─ Pipeline 2 ← Thread 2 处理请求 B  (共享 GPU)
├─ Pipeline 3 ← Thread 3 处理请求 C
└─ Pipeline 4 ← Thread 4 处理请求 D

GPU 调度器：多个线程共享 GPU 时间片
```

### ⚠️ 注意事项

| 方面 | 说明 |
|------|------|
| **显存限制** | 所有 Pipeline 共享同一块 GPU 的显存 |
| **计算限制** | GPU 同一时间只能处理一个计算任务，但调度很快 |
| **最佳场景** | 大量小请求，GPU 空闲时间多 |
| **最坏场景** | 超大图片，单个 Pipeline 已占满 GPU |

---

## 🚀 使用方式

### 1. 单 GPU，多个 Pipeline

```python
from medical_ocr.pipeline_pool_v2 import create_pipeline_pool

# 创建 4 个 Pipeline，都在 GPU 0 上
pool = create_pipeline_pool(
    pipeline_config=config.pipeline,
    pool_size=4,
    yolo_model_dir="/path/to/yolo",
    uvdoc_model_dir="/path/to/uvdoc",
    mode="single_gpu",            # 关键：单 GPU 模式
    gpu_device_id=0,              # 使用第 0 块 GPU
    gpu_memory_fraction=0.9       # 限制显存（可选）
)

pool.initialize()

# 处理请求（自动调度到空闲 Pipeline）
results = pool.process(request_data)

# 批量并行处理（4 个请求同时处理）
batch_results = pool.process_batch([req1, req2, req3, req4])

pool.shutdown()
```

### 2. 显存控制（重要！）

如果显存不够用，限制每个 Pipeline 的显存：

```python
pool = create_pipeline_pool(
    ...,
    gpu_memory_fraction=0.7  # 总共用 70% 的显存
)

# 或者每个 Pipeline 限制：
# 4 个 Pipeline，每个限制 20% 显存 = 总共 80%
```

### 3. 性能对比

| 配置 | 并行度 | 8个请求耗时 | 说明 |
|------|--------|------------|------|
| 1 Pipeline | 1 | 8.0s | 串行 |
| 2 Pipelines | 2 | 4.2s | ~1.9x 加速 |
| 4 Pipelines | 4 | 2.1s | ~3.8x 加速 |
| 8 Pipelines | 8 | 1.8s | ~4.4x 加速 (瓶颈在GPU) |

---

## 📊 两种模式对比

| 特性 | 单 GPU 多 Pipeline | 多 GPU 多 Pipeline |
|------|-------------------|-------------------|
| **硬件需求** | 1 块 GPU | 多块 GPU |
| **并行库** | `ThreadPoolExecutor` | `ProcessPoolExecutor` |
| **显存** | 共享 | 每个 GPU 独立 |
| **最佳场景** | 小请求，高吞吐 | 大请求，GPU 密集 |
| **GIL 影响** | 很小 (GPU 计算时释放) | 无 (每个进程独立 GIL) |

### 选择建议：

```python
# 小图片，高并发 → 单 GPU 多 Pipeline
pool = create_pipeline_pool(..., mode="single_gpu", pool_size=4 or 8)

# 大图片，GPU 密集 → 多 GPU（每个 GPU 1-2 个 Pipeline）
pool = create_pipeline_pool(..., mode="multi_gpu", gpu_device_ids=[0, 1])
```

---

## 🏗️ 架构细节

### SingleGPUPipelinePool

```python
class SingleGPUPipelinePool:
    # 内部使用 ThreadPoolExecutor
    # 多个 Pipeline 实例
    # 所有线程共享同一块 GPU
    # GPU 调度器负责分配时间
```

### 工作流程

```
1. 请求进入
2. 获取空闲 Pipeline（从 queue）
3. 提交任务到线程池
4. GPU 上执行计算（调度器分配）
5. 完成，返回 Pipeline 到 pool
6. 下一个请求使用
```

---

## 🎯 最佳实践

### Pool 大小建议

| GPU 显存 | 推荐 Pool Size |
|---------|--------------|
| 8GB | 2-4 |
| 12GB | 4-6 |
| 24GB | 6-8 |
| 48GB+ | 8-16 |

### 监控显存

```python
import torch

# 获取当前 GPU 显存使用
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"GPU Memory Used: {allocated:.2f}GB")
```

---

## 📁 相关文件

| 文件 | 说明 |
|------|------|
| `pipeline_pool_v2.py` | 核心实现（单/多 GPU 都支持）|
| `examples/single_gpu_demo.py` | 单 GPU 使用示例 |
| `medical_ocr/high_perf_server_v2.py` | 服务器示例 |

---

## 💡 关键要点

✅ **可以！单卡可以跑多个 Pipeline**

✅ **使用 `ThreadPoolExecutor`，不是 `ProcessPoolExecutor`**

✅ **显存是共享的，注意不要 OOM**

✅ **通过提高吞吐量获得性能提升，而不是真正的并行**

✅ **适合场景：小图片、高并发**
