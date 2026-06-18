# 预处理与后处理微服务 Spec

## Why
当前 GLM-OCR 服务缺少文档预处理（文档检测、方向矫正、扭曲矫正）和后处理能力。需要创建独立的微服务，支持串行处理流水线，既能独立部署，也能与现有 OCR 服务灵活组合。

## What Changes
- 新增 `preprocess/` 目录，包含独立的预处理微服务
- 实现三个串行预处理步骤：文档检测 → 方向矫正 → 扭曲矫正
- 新增 `postprocess/` 目录，包含后处理微服务
- 提供预处理客户端，可集成到 AsyncPipeline
- 支持独立部署和组合部署两种模式

## Impact
- Affected specs: glmocr-async-service（扩展）
- Affected code:
  - `preprocess/` — 新增预处理微服务目录
  - `postprocess/` — 新增后处理微服务目录
  - `glmocr/preprocess_client.py` — 预处理客户端
  - `glmocr/postprocess_client.py` — 后处理客户端
  - `glmocr/async_pipeline.py` — 集成预处理和后处理

## ADDED Requirements

### Requirement: 预处理微服务
系统 SHALL 提供独立的预处理微服务，包含文档检测、方向矫正、扭曲矫正三个串行步骤。

#### Scenario: 独立部署预处理服务
- **WHEN** 用户启动预处理服务
- **THEN** 服务监听指定端口，提供 `/preprocess` 端点

#### Scenario: 串行处理流水线
- **WHEN** 上传图片到 `/preprocess`
- **THEN** 依次执行：文档检测 → 方向矫正 → 扭曲矫正
- **AND** 返回处理后的图片和中间结果

#### Scenario: 单步骤调用
- **WHEN** 调用 `/detect`、`/orient`、`/dewarp` 端点
- **THEN** 只执行对应的单一步骤

### Requirement: 后处理微服务
系统 SHALL 提供独立的后处理微服务，对 OCR 结果进行特定处理。

#### Scenario: 后处理结果
- **WHEN** 提交 OCR 结果到 `/postprocess`
- **THEN** 执行自定义后处理逻辑
- **AND** 返回处理后的结果

### Requirement: 预处理客户端
系统 SHALL 提供预处理客户端，可集成到 AsyncPipeline。

#### Scenario: 集成到 Pipeline
- **WHEN** AsyncPipeline 配置了预处理服务
- **THEN** 在布局检测前自动调用预处理
- **AND** 使用预处理后的图片进行后续处理

#### Scenario: 独立使用
- **WHEN** 用户单独使用预处理客户端
- **THEN** 可以独立调用预处理服务

### Requirement: 灵活部署模式
系统 SHALL 支持独立部署和组合部署两种模式。

#### Scenario: 独立部署
- **WHEN** 使用 `docker-compose.preprocess.yml`
- **THEN** 只启动预处理服务

#### Scenario: 组合部署
- **WHEN** 使用 `docker-compose.full.yml`
- **THEN** 启动预处理 + OCR + 后处理完整服务

## MODIFIED Requirements

### Requirement: AsyncPipeline 配置
PipelineConfig SHALL 新增 `preprocess` 和 `postprocess` 配置段。

### Requirement: 配置文件
config.yaml SHALL 包含预处理和后处理的配置示例。

## REMOVED Requirements
无（新增功能）
