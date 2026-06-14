# Checklist

- [ ] Redis 依赖已添加到 `apps/backend/pyproject.toml`
- [ ] `Settings` 类中包含 `REDIS_URL` 和 `REDIS_ENABLED` 配置项
- [ ] `apps/backend/app/db/redis.py` 文件已创建
- [ ] `init_redis()` 函数实现且能正确初始化异步连接池
- [ ] `close_redis()` 函数实现且能正确关闭连接池
- [ ] `get_redis()` 函数实现且返回 Redis 连接实例
- [ ] Redis 不可用时应用降级运行，不抛出未捕获异常
- [ ] `apps/backend/app/main.py` 的 `lifespan()` 中调用了 `init_redis()`
- [ ] `apps/backend/app/main.py` 的 `lifespan()` 中调用了 `close_redis()`
- [ ] `apps/backend/app/core/aggregator.py` 文件已创建
- [ ] `RegionAggregator` 类已实现
- [ ] `register_document(doc_id, region_count)` 方法实现且正确写入 Redis Hash
- [ ] `on_region_complete(doc_id, region_id, result)` 方法实现且能检测所有 region 是否完成
- [ ] `get_document_results(doc_id)` 方法实现且返回完整的聚合结果
- [ ] `check_timeout(doc_id, timeout_seconds)` 方法实现且能标记超时 region
- [ ] `layout_ocr.py` 中布局检测完成后调用了 `aggregator.register_document()`
- [ ] 所有新增代码无语法错误，可正常导入
- [ ] Redis 连接参数可通过环境变量覆盖
