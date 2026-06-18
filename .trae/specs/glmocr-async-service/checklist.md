# Checklist

## Redis 配置集成
- [x] `RedisConfig` 数据类已创建并包含在 `glmocr/config.py` 中
- [x] `PipelineConfig` 已添加 `redis: RedisConfig` 字段
- [x] `config.yaml` 已更新，包含 redis 配置段示例
- [x] 环境变量 `GLMOCR_REDIS_URL` 支持已实现

## 结果聚合器实现
- [x] `glmocr/aggregator.py` 文件已创建
- [x] `RegionAggregator` 类已实现
- [x] `register_document()` 方法已实现
- [x] `on_region_complete()` 方法已实现
- [x] `get_progress()` 方法已实现
- [x] `is_complete()` 方法已实现
- [x] `get_result()` 方法已实现
- [x] Redis 异步客户端连接池已实现
- [x] 结果聚合逻辑使用 Redis Hash 存储
- [x] 原子操作递增完成计数已实现

## 异步 Pipeline 支持
- [x] `process_async()` 方法已添加到 `pipeline.py`
- [x] Fire-and-forget 提交逻辑已实现
- [x] Region 结果回调机制已实现
- [x] 原有 `process()` 方法保持不变

## 异步服务实现
- [x] `glmocr/async_server.py` 文件已创建
- [x] FastAPI 应用已初始化
- [x] `/parse/async` POST 端点已实现
- [x] `/parse/status/{task_id}` GET 端点已实现
- [x] `/ws/{task_id}` WebSocket 端点已实现
- [x] WebSocket 支持进度推送和最终结果推送

## 集成测试
- [x] `RegionAggregator` 单元测试已编写
- [x] Redis 连接失败降级处理已测试
- [x] 完整异步流程集成测试已通过
- [x] WebSocket 推送测试已通过
- [x] 向后兼容性验证已通过

## 文档和示例
- [x] README 已更新，包含异步服务使用说明
- [x] 启动脚本示例已提供
- [x] 客户端调用示例代码已提供
