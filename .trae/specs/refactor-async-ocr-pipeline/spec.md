# 异步 OCR Client 和 Fire-and-Forget Pipeline 改造 Spec

## Why
当前 OCR Client 使用同步的 `requests` 库，虽然通过 `run_in_executor` 实现了异步调用，但存在以下问题：
1. 线程池资源浪费：每个 region 占用一个线程
2. 连接池管理复杂：需要维护较大的连接池
3. 无法充分利用异步 I/O 的优势

需要改造为真正的异步架构，使用 `httpx` 或 `aiohttp` 实现异步 HTTP 客户端，并让 Pipeline 完全采用 fire-and-forget 模式。

## What Changes
- **BREAKING**: 新增 `AsyncOCRClient` 类，使用 `httpx.AsyncClient` 替代同步的 `OCRClient`
- 新增 `AsyncPipeline` 类，完全基于 asyncio 实现，不再使用线程池
- 修改 `async_server.py` 使用新的异步组件
- 保持原有 `OCRClient` 和 `Pipeline` 不变，确保向后兼容

## Impact
- Affected specs: glmocr-async-service（扩展）
- Affected code:
  - `glmocr/async_ocr_client.py` — 新增异步 OCR 客户端
  - `glmocr/async_pipeline.py` — 新增完全异步的 Pipeline
  - `glmocr/async_server.py` — 使用新的异步组件
  - `glmocr/config.py` — 新增异步相关配置

## ADDED Requirements

### Requirement: 异步 OCR 客户端
系统 SHALL 提供 `AsyncOCRClient`，使用 `httpx.AsyncClient` 实现真正的异步 HTTP 请求。

#### Scenario: 异步提交单个 region
- **WHEN** 调用 `async_process(request_data)`
- **THEN** 返回 `asyncio.Task`，不阻塞事件循环

#### Scenario: 批量并发提交
- **WHEN** 调用 `async_process_batch(requests)` 提交多个 region
- **THEN** 所有请求并发执行，返回任务列表

#### Scenario: 连接池管理
- **WHEN** 创建 `AsyncOCRClient` 实例
- **THEN** 自动创建 `httpx.AsyncClient` 连接池，支持配置最大连接数

### Requirement: 完全异步 Pipeline
系统 SHALL 提供 `AsyncPipeline`，完全基于 asyncio 实现，不使用线程池。

#### Scenario: 异步处理文档
- **WHEN** 调用 `process_async(request_data, doc_id, aggregator)`
- **THEN** 布局检测在后台线程执行（CPU 密集），OCR 提交完全异步

#### Scenario: Fire-and-forget 模式
- **WHEN** 布局检测完成，产生多个 region
- **THEN** 立即返回 `doc_id`，所有 region 在后台并发处理

#### Scenario: 结果回调
- **WHEN** 某个 region 的 OCR 完成
- **THEN** 立即调用 `aggregator.on_region_complete()`，无需等待其他 region

### Requirement: 异步配置
系统 SHALL 支持异步相关的配置项。

#### Scenario: 异步客户端配置
- **WHEN** 配置 `async_ocr.max_connections`
- **THEN** `AsyncOCRClient` 使用该值作为最大连接数

#### Scenario: 并发控制
- **WHEN** 配置 `async_pipeline.max_concurrent_regions`
- **THEN** 同时处理的 region 数量不超过该值（使用 `asyncio.Semaphore`）

## MODIFIED Requirements

### Requirement: 异步服务
`async_server.py` SHALL 使用 `AsyncPipeline` 和 `AsyncOCRClient` 替代原有组件。

### Requirement: Pipeline 配置
`PipelineConfig` SHALL 新增 `async_ocr` 和 `async_pipeline` 配置段。

## REMOVED Requirements
无（保持向后兼容）
