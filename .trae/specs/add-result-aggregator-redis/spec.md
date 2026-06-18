# 结果聚合器 + Redis 集成 Spec

## Why
当前 GLM-OCR 的 pipeline 是同步阻塞式架构，Worker 必须等待 vLLM 返回所有 region 结果后才能继续。为了实现异步流水线（fire-and-forget），需要先引入结果聚合器，用 Redis 跟踪每个文档的 region 完成情况，为后续异步改造奠定基础。

## What Changes
- 新增 Redis 依赖（redis-py + aioredis）
- 新增 `RegionAggregator` 组件，负责跟踪每个文档的 region 完成情况
- 新增 Redis 连接管理模块
- 在配置中新增 Redis 相关配置项
- 在应用生命周期中初始化/关闭 Redis 连接
- 修改 `layout_ocr.py` 步骤，在提交 region 到 vLLM 时注册 region 总数到 Redis
- 新增聚合触发机制：当某个文档的所有 region 都处理完成时，触发结果合并

## Impact
- Affected specs: 无（新增功能）
- Affected code:
  - `apps/backend/app/utils/config.py` — 新增 Redis 配置
  - `apps/backend/app/db/database.py` 或新增 `app/db/redis.py` — Redis 连接管理
  - `apps/backend/app/main.py` — 生命周期中初始化 Redis
  - `apps/backend/app/core/aggregator.py` — 新增结果聚合器
  - `apps/backend/app/core/steps/layout_ocr.py` — 提交 region 时注册到聚合器
  - `apps/backend/pyproject.toml` — 新增 redis 依赖

## ADDED Requirements

### Requirement: Redis 连接管理
系统 SHALL 提供 Redis 连接管理模块，支持异步连接池，在应用启动时初始化、关闭时释放。

#### Scenario: Redis 连接初始化成功
- **WHEN** 应用启动且 Redis 配置正确
- **THEN** Redis 连接池创建成功，可通过 `get_redis()` 获取连接实例

#### Scenario: Redis 连接失败
- **WHEN** Redis 服务不可用
- **THEN** 应用仍能启动（降级模式），聚合器功能不可用，日志记录警告

### Requirement: 结果聚合器
系统 SHALL 提供 `RegionAggregator`，能够跟踪每个文档的 region 完成情况，并在所有 region 完成后触发合并。

#### Scenario: 注册文档的 region 列表
- **WHEN** Pipeline 完成布局检测，将文档的 N 个 region 提交到 vLLM
- **THEN** 聚合器在 Redis 中记录该文档有 N 个 region 待处理

#### Scenario: 单个 region 完成
- **WHEN** vLLM 返回某个 region 的 OCR 结果
- **THEN** 聚合器将该 region 结果存入 Redis Hash，并更新已完成计数

#### Scenario: 所有 region 完成
- **WHEN** 某文档的所有 region 都已完成
- **THEN** 聚合器触发结果合并回调，将所有 region 结果整合为完整文档结果

#### Scenario: region 处理超时
- **WHEN** 某文档的 region 在指定时间内未全部完成
- **THEN** 聚合器标记超时 region 为失败，并触发合并（包含失败标记）

### Requirement: Redis 配置
系统 SHALL 支持通过环境变量配置 Redis 连接参数。

#### Scenario: 默认配置
- **WHEN** 未提供 Redis 环境变量
- **THEN** 使用默认配置：`redis://localhost:6379/0`

#### Scenario: 自定义配置
- **WHEN** 设置环境变量 `REDIS_URL=redis://custom:6380/1`
- **THEN** 使用自定义配置连接 Redis

## MODIFIED Requirements

### Requirement: 应用生命周期管理
应用启动时 SHALL 初始化 Redis 连接池，关闭时释放连接。Redis 不可用时应用降级运行。

### Requirement: layout_ocr 步骤
布局检测完成后 SHALL 将 region 信息注册到聚合器，记录待处理的 region 总数。
