# Tasks

- [ ] Task 1: 新增 Redis 依赖
  - [ ] SubTask 1.1: 在 `apps/backend/pyproject.toml` 中添加 `redis[hiredis]>=5.0.0` 依赖

- [ ] Task 2: 新增 Redis 配置项
  - [ ] SubTask 2.1: 在 `apps/backend/app/utils/config.py` 的 `Settings` 类中新增 `REDIS_URL: str = "redis://localhost:6379/0"` 和 `REDIS_ENABLED: bool = True` 配置

- [ ] Task 3: 新增 Redis 连接管理模块
  - [ ] SubTask 3.1: 创建 `apps/backend/app/db/redis.py`，实现异步 Redis 连接池的初始化（`init_redis()`）、关闭（`close_redis()`）和获取实例（`get_redis()`）
  - [ ] SubTask 3.2: 实现降级逻辑：Redis 不可用时记录警告日志，不阻塞应用启动

- [ ] Task 4: 集成 Redis 到应用生命周期
  - [ ] SubTask 4.1: 在 `apps/backend/app/main.py` 的 `lifespan()` 中调用 `init_redis()` 和 `close_redis()`

- [ ] Task 5: 实现结果聚合器
  - [ ] SubTask 5.1: 创建 `apps/backend/app/core/aggregator.py`，实现 `RegionAggregator` 类
  - [ ] SubTask 5.2: 实现 `register_document(doc_id, region_count)` — 在 Redis Hash 中记录文档的 region 总数
  - [ ] SubTask 5.3: 实现 `on_region_complete(doc_id, region_id, result)` — 存储单个 region 结果并检查是否全部完成
  - [ ] SubTask 5.4: 实现 `get_document_results(doc_id)` — 获取某文档所有 region 的聚合结果
  - [ ] SubTask 5.5: 实现 `check_timeout(doc_id, timeout_seconds)` — 检查并处理超时 region

- [ ] Task 6: 修改 layout_ocr 步骤集成聚合器
  - [ ] SubTask 6.1: 在 `apps/backend/app/core/steps/layout_ocr.py` 的 `_call_ocr_service()` 中，布局检测完成后调用 `aggregator.register_document()` 注册 region 总数

- [ ] Task 7: 验证
  - [ ] SubTask 7.1: 验证 Redis 连接管理模块可正常初始化和关闭
  - [ ] SubTask 7.2: 验证聚合器的 register / on_region_complete / get_document_results 流程正确
  - [ ] SubTask 7.3: 验证 Redis 不可用时应用降级运行不报错

# Task Dependencies
- [Task 3] depends on [Task 1, Task 2]
- [Task 4] depends on [Task 3]
- [Task 5] depends on [Task 3]
- [Task 6] depends on [Task 5]
- [Task 7] depends on [Task 4, Task 5, Task 6]
