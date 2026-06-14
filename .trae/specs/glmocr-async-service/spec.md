# 基于 GLM-OCR SDK 的异步流水线服务 Spec

## Why
用户希望基于 `glmocr` SDK（而非官方 demo `apps/backend`）构建一个异步 OCR 服务。当前 SDK 的 pipeline 使用多线程但仍是同步阻塞模式，需要改造为 fire-and-forget 异步模式，实现预处理、布局检测、OCR 推理的解耦，提高吞吐量。

## What Changes
- 新增 Redis 依赖到 `glmocr` 项目
- 在 `glmocr/config.py` 中新增 Redis 相关配置
- 新增 `glmocr/aggregator.py` 结果聚合器模块
- 新增 `glmocr/async_server.py` 异步服务（基于 FastAPI）
- 修改 `glmocr/pipeline/pipeline.py` 支持异步提交模式
- 新增 WebSocket/SSE 端点用于推送 OCR 结果

## Impact
- Affected specs: 无（全新服务，不修改原有 server.py）
- Affected code:
  - `glmocr/config.py` — 新增 Redis 配置
  - `glmocr/aggregator.py` — 新增结果聚合器
  - `glmocr/async_server.py` — 新增异步服务入口
  - `glmocr/pipeline/pipeline.py` — 新增异步处理方法

## ADDED Requirements

### Requirement: Redis 配置
系统 SHALL 支持通过环境变量或 YAML 配置 Redis 连接参数。

#### Scenario: 默认配置
- **WHEN** 未提供 Redis 配置
- **THEN** 使用默认值：`redis://localhost:6379/0`

#### Scenario: 环境变量配置
- **WHEN** 设置 `GLMOCR_REDIS_URL=redis://custom:6380/1`
- **THEN** 使用自定义 Redis 地址

### Requirement: 结果聚合器
系统 SHALL 提供 `RegionAggregator`，能够跟踪每个文档的 region 完成情况。

#### Scenario: 注册文档
- **WHEN** Pipeline 完成布局检测，检测到 N 个 region
- **THEN** 聚合器在 Redis 中记录 `doc:{doc_id}:total_regions = N`

#### Scenario: Region 完成
- **WHEN** vLLM 返回某个 region 的 OCR 结果
- **THEN** 聚合器存储结果到 `doc:{doc_id}:region:{region_id}`，并递增完成计数

#### Scenario: 全部完成
- **WHEN** 某文档所有 region 都已完成
- **THEN** 聚合器触发回调，返回完整的文档 OCR 结果

### Requirement: 异步服务
系统 SHALL 提供异步 HTTP 服务，支持提交文档后立即返回，通过 WebSocket 推送结果。

#### Scenario: 提交文档
- **WHEN** 客户端 POST 到 `/parse/async`
- **THEN** 服务立即返回 `task_id`，后台开始处理

#### Scenario: 查询状态
- **WHEN** 客户端 GET `/parse/status/{task_id}`
- **THEN** 返回当前进度（已完成的 region 数 / 总 region 数）

#### Scenario: WebSocket 推送
- **WHEN** 客户端连接 WebSocket `/ws/{task_id}`
- **THEN** 每个 region 完成时推送进度，全部完成时推送最终结果

### Requirement: 异步 Pipeline
Pipeline SHALL 支持异步处理模式，布局检测后立即提交 region 到 vLLM，不等待返回。

#### Scenario: Fire-and-forget 提交
- **WHEN** 布局检测完成，产生多个 region
- **THEN** 每个 region 异步提交到 vLLM，Pipeline 立即处理下一张图片

#### Scenario: 结果回调
- **WHEN** vLLM 返回某个 region 的结果
- **THEN** 调用聚合器的 `on_region_complete()` 方法

## MODIFIED Requirements

### Requirement: Pipeline 配置
PipelineConfig SHALL 新增 `redis` 和 `async_mode` 配置项。

## REMOVED Requirements
无（不修改原有功能，仅新增）
