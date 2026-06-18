# Tasks

## 1. Redis 配置集成
- [x] 1.1 在 `glmocr/config.py` 中新增 `RedisConfig` 数据类
  - 包含 `url: str = "redis://localhost:6379/0"`
  - 包含 `key_prefix: str = "glmocr"`
  - 包含 `max_connections: int = 10`
- [x] 1.2 在 `PipelineConfig` 中添加 `redis: RedisConfig` 字段
- [x] 1.3 更新 `config.yaml` 示例，添加 redis 配置段
- [x] 1.4 添加环境变量支持：`GLMOCR_REDIS_URL`

## 2. 结果聚合器实现
- [x] 2.1 创建 `glmocr/aggregator.py` 文件
- [x] 2.2 实现 `RegionAggregator` 类
  - `__init__(self, redis_client, key_prefix)`
  - `register_document(doc_id, total_regions)` - 注册文档及 region 总数
  - `on_region_complete(doc_id, region_id, result)` - 记录单个 region 完成
  - `get_progress(doc_id)` - 获取进度信息
  - `is_complete(doc_id)` - 检查是否全部完成
  - `get_result(doc_id)` - 获取完整结果
- [x] 2.3 实现 Redis 连接管理
  - 使用 `redis.asyncio` 异步客户端
  - 连接池复用
  - 自动重连机制
- [x] 2.4 实现结果聚合逻辑
  - 使用 Redis Hash 存储 region 结果
  - 使用原子操作递增完成计数
  - 全部完成时触发回调

## 3. 异步 Pipeline 支持
- [x] 3.1 在 `glmocr/pipeline/pipeline.py` 中新增 `process_async` 方法
  - 接收 `doc_id` 参数
  - 布局检测后立即提交所有 region 到 vLLM
  - 返回 `task_id` 而不阻塞等待结果
- [x] 3.2 实现 region 结果回调机制
  - vLLM 返回结果时调用 `aggregator.on_region_complete()`
  - 支持并发处理多个 region
- [x] 3.3 保持原有 `process()` 方法不变，确保向后兼容

## 4. 异步服务实现
- [x] 4.1 创建 `glmocr/async_server.py`
  - 使用 FastAPI 框架
  - 复用现有 `Pipeline` 实例
- [x] 4.2 实现 `/parse/async` POST 端点
  - 接收图片/PDF 文件
  - 调用 `pipeline.process_async()`
  - 返回 `{"task_id": "xxx"}`
- [x] 4.3 实现 `/parse/status/{task_id}` GET 端点
  - 查询聚合器获取进度
  - 返回 `{"completed": 5, "total": 10, "status": "processing"}`
- [x] 4.4 实现 `/ws/{task_id}` WebSocket 端点
  - 连接时检查任务是否存在
  - 每个 region 完成时推送进度消息
  - 全部完成时推送最终结果
  - 支持客户端断开重连

## 5. 集成测试
- [x] 5.1 编写单元测试
  - 测试 `RegionAggregator` 的各个方法
  - 测试 Redis 连接失败时的降级处理
- [x] 5.2 编写集成测试
  - 测试完整的异步提交流程
  - 测试 WebSocket 推送
- [x] 5.3 验证向后兼容性
  - 确保原有 `server.py` 仍可正常运行
  - 确保 `process()` 方法不受影响

## 6. 文档和示例
- [x] 6.1 更新 README，添加异步服务使用说明
- [x] 6.2 提供启动脚本示例
- [x] 6.3 提供客户端调用示例代码
