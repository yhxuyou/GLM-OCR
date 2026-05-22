# ══════════════════════════════════════════════════════════════════════
# Nginx 限流配置速查表
# ══════════════════════════════════════════════════════════════════════

# ── 限流区域定义 (已在 nginx.conf 中配置) ───────────────────────────
#
# 区域                     Key               速率            用途
# ──────────────────────────────────────────────────────────────────────
# global                   $server_name      500 r/s         全局总限流
# per_ip                   $binary_remote_addr 20 r/s        单 IP 限流
# per_ip_burst             $binary_remote_addr 60 r/s        单 IP 突发
# per_apikey               $http_authorization 100 r/s       用户级别
# conn_per_ip              $binary_remote_addr 10 连接       并发连接数

# ── 各 API 端点的限流阈值 ──────────────────────────────────────────
#
# 端点                    限流规则               突发            说明
# ──────────────────────────────────────────────────────────────────────
# /api/                   global(500) + per_ip(10)  500/10  后端 API
# /glmocr/                per_ip(5) + per_apikey(20) 5/20  推理服务 (最贵)
# /health                 per_ip(20)              20       健康检查
# /metrics                仅内网                               监控指标
# /docs                   per_ip(5)               5         API 文档

# ── 熔断参数 ─────────────────────────────────────────────────────────
#
# 参数                    值            说明
# ──────────────────────────────────────────────────────────────────────
# max_fails                3            连续失败 3 次认为节点不可用
# fail_timeout             30s          标记不可用后等待 30s 再恢复
# proxy_next_upstream      error|timeout|invalid_header|http_500|502|503
# proxy_next_upstream_tries 3           最多尝试 3 个上游节点
# proxy_next_upstream_timeout 30s       30s 内未完成则放弃

# ── 关键参数调优指南 ─────────────────────────────────────────────────

# 1. limit_req zone=per_ip burst=N
#    burst: 突发缓冲区大小
#    公式: burst = rate × 允许的突发秒数
#    例如: rate=20r/s, 允许 2s 突发 → burst=40
#    但 GLM-OCR 单个请求较长, burst 应较小

# 2. nodelay vs delay
#    nodelay: 允许立即处理 burst 内的请求, 不延迟
#    delay=N: 前 N 个请求立即处理, 后续排队
#    delay 参数: limit_req zone=per_ip burst=10 nodelay delay=5
#    - 前 5 个请求: 立即处理 (不延迟)
#    - 后 5 个请求: 立即处理, 但会计入速率 (超出部分返回 429)
#    - 推荐 /glmocr/ 使用 delay=2 (允许小突发)

# 3. proxy_next_upstream 最佳实践
#    GLM-OCR Pipeline 建议的配置:
#    proxy_next_upstream error timeout http_500 http_502 http_503;
#    - error:          连接/读/写超时 → 尝试下一个
#    - timeout:        超时 → 尝试下一个
#    - http_500:       服务端错误 → 尝试下一个
#    - http_502/503:   Bad Gateway / 服务不可用 → 尝试下一个
#    - 不包含 http_429: 限流不应该触发重试（会加重负载）

# ── 灰度发布流量比例建议 ──────────────────────────────────────────
#
# 阶段         稳定版本  灰度版本  分流规则                    验证方式
# ──────────────────────────────────────────────────────────────────────
# 内部测试      100%      0%       IP 在白名单内走灰度        内网 IP
# 小流量        95%       5%       Cookie canary=canary       QA 团队
# 灰度放量      80%       20%      10% 随机用户               Header 标记
# 全量发布      0%        100%     全部切换                   监控回滚
# 回滚          100%      0%       清除灰度标记               立即生效

# ── 性能预期 (基于 Nginx + T4) ──────────────────────────────────────

# 无 Nginx 直连:
#   后端 API:     ~5000 QPS (无状态请求)
#   Pipeline:     ~200  QPS (GPU 推理瓶颈)
#
# 有 Nginx 限流:
#   后端 API:     限流 500 QPS + 单 IP 20 QPS
#   Pipeline:     限流单 IP 5 QPS (保护 GPU)
#   总吞吐:       受 GPU 限制, Nginx 层不构成瓶颈
#
# Nginx 本身:
#   单实例:       ~50000 QPS (纯转发, 无 TLS)
#   带 TLS:       ~20000 QPS (AES-NI 加速)
#   带限流:       ~30000 QPS (限流规则有一定开销)