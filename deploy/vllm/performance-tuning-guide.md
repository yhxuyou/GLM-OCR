# ── GLM-OCR 性能调优速查表 ─────────────────────────────────────────────
# 针对不同 GPU 的推荐配置
# ──────────────────────────────────────────────────────────────────────
#
# GPU 型号    显存     TP  max_seqs  quant   预期 QPS  场景
# ──────────────────────────────────────────────────────────────────────
# T4          16 GB    1    256       awq     50-80     低成本推理
# T4          16 GB    1    128       FP16    30-50     标准
# V100-32G    32 GB    1    512       awq     80-120    中等负载
# V100-32G    32 GB    1    256       FP16    50-80     高质量
# A10         24 GB    1    512       awq     100-150   性价比之选
# A100-40G    40 GB    1    1024      awq     150-250   高负载
# A100-80G    80 GB    1    2048      FP16    200-300   极致吞吐
# A100-80G    80 GB    2    4096      FP16    300-500   双卡并行
# H100-80G    80 GB    1    2048      FP8     400-600   最新架构
# ──────────────────────────────────────────────────────────────────────

# ── 不同场景的配置建议 ─────────────────────────────────────────────────

# 场景 1: 成本敏感 (T4 16G, AWQ 量化)
# ──────────────────────────────────────────────────────────────────────
# gpu_memory_utilization: 0.95  # 最大化利用有限显存
# max_num_seqs: 256             # 配合 Continuous Batching
# quantization: awq             # INT4 节省 75% 显存
# max_model_len: 4096           # 适当减小以容纳更多序列
# enable_chunked_prefill: true
# num_scheduler_steps: 8

# 场景 2: 均衡配置 (A100-80G, FP16)
# ──────────────────────────────────────────────────────────────────────
# gpu_memory_utilization: 0.90
# max_num_seqs: 512
# max_model_len: 8192
# quantization: ""              # FP16 保持最高精度
# enable_chunked_prefill: true
# num_scheduler_steps: 4

# 场景 3: 极致吞吐 (A100-80G × 2, TP=2)
# ──────────────────────────────────────────────────────────────────────
# tensor_parallel_size: 2
# gpu_memory_utilization: 0.90
# max_num_seqs: 2048            # 双卡大幅提升并发
# max_model_len: 8192
# quantization: ""
# enable_chunked_prefill: true
# num_scheduler_steps: 16       # 增加调度步骤提升吞吐

# 场景 4: 低延迟 (响应时间优先)
# ──────────────────────────────────────────────────────────────────────
# max_num_seqs: 32              # 减少队列等待
# max_model_len: 4096
# scheduling: sync              # 同步调度, 减少调度延迟
# enable_chunked_prefill: false
# gpu_memory_utilization: 0.85

# ── 关键参数调优指南 ──────────────────────────────────────────────────

# 1. max_num_seqs 调优
#    - 增大 → 吞吐↑ 延迟↑ (排队时间长)
#    - 减小 → 吞吐↓ 延迟↓ (适合实时场景)
#    - 公式: 目标 QPS × 平均推理时间 = 所需并行度
#    - 例如: 100 QPS × 0.5s = 50 并发 → max_num_seqs=64

# 2. gpu_memory_utilization 调优
#    - 增大 → 更多 KV Cache 空间 → 支持更多并发
#    - 但留太少 → CUDA OOM (Out Of Memory)
#    - 安全值: 0.85-0.90, 激进值: 0.95

# 3. enable_chunked_prefill
#    - true:  解决 Block LLM 问题, 提高长文档推理效率
#    - false: 减少调度开销, 适合短文本场景
#    - GLM-OCR 场景推荐开启 (文档通常较长)

# 4. num_scheduler_steps
#    - 增大 → 每次调度处理更多请求 → 吞吐↑
#    - 但过大 → 首 token 延迟↑
#    - 推荐值: 4-16

# ── 性能基准测试 ────────────────────────────────────────────────────

# 使用 vLLM 自带的 benchmark 工具:
#   python -m vllm.benchmarks.benchmark_serving \
#     --backend vllm \
#     --model THUDM/glm-ocr \
#     --dataset-name sharegpt \
#     --num-prompts 1000 \
#     --request-rate 10 \
#     --port 8000

# 输出解读:
#   Request throughput:    每秒完成的请求数
#   Average latency:       平均响应时间
#   P99 latency:           99% 请求的响应时间 (SLA 指标)
#   TTFT (Time to First Token): 首 token 延迟
#   TPOT (Time per Output Token): 每个输出 token 的时间