# 医疗票据异步解析服务 Spec

## Why

现有 `glmocr/server.py` 是同步 Flask 接口，解析一张医疗票据就会阻塞整个 worker 线程，吞吐量受限于并发请求数 × 单张解析时延。医疗票据场景常见一次上传多张（住院/门诊/检查/药品），需要把"提交"和"取结果"解耦成异步任务，并复用我们刚搭好的 `PreprocessPool`（4 进程预处理 + 异步 OCR + 按文档聚合）。同时医疗票据需要从 OCR 文本进一步抽取出结构化字段（票据号/金额/项目明细等），便于直接入库/对账。

## What Changes

- **新增** `glmocr/medical_extractor.py` — 医疗票据字段抽取器，把 OCR 出的 markdown 送给同一个 vLLM 后端做结构化抽取，输出 Pydantic `BillRecord`。
- **新增** `glmocr/medical_server.py` — FastAPI 异步服务，参考 `server.py` 的接口风格（POST 提交 + GET 拉取），复用 `PreprocessPool` 作为后台引擎。
- **新增** `glmocr/config.py` 的 `MedicalServerConfig` 与 `MedicalExtractorConfig` 配置块；`GlmOcrConfig` 增加 `medical_server` 字段。
- **新增** `glmocr/config.yaml` 对应配置段。
- **新增** `glmocr/medical_extractor.py`、`glmocr/medical_server.py` 的单测。
- 不修改 `glmocr/server.py`（保留向后兼容）。
- 不修改 `PreprocessPool` 本身。

## Impact

- **Affected specs**：无既有 spec；这是新能力。
- **Affected code**：
  - 新增：`glmocr/medical_extractor.py`、`glmocr/medical_server.py`、`glmocr/medical_extractor/__init__.py`（如有）
  - 扩展：`glmocr/config.py`、`glmocr/config.yaml`
  - 测试：`glmocr/tests/test_medical_extractor.py`、`glmocr/tests/test_medical_server.py`
- **新依赖**：`fastapi`、`uvicorn`、`python-multipart`（文件上传用）；抽取阶段复用 `OCRClient`，不引入额外 LLM SDK。
- **运行方式**：`python -m glmocr.medical_server --config glmocr/config.yaml --port 8080`（设计入参，最终以实现为准）。

## ADDED Requirements

### Requirement: 异步提交接口

`POST /v1/bills/parse` 必须异步接收 1～N 张医疗票据图片，立即返回 `task_id`（HTTP 202），不阻塞调用方。

#### Scenario: 提交单张 base64 图片
- **WHEN** 客户端 POST `application/json`，body 为 `{"image": "data:image/png;base64,..."}`
- **THEN** 接口立刻返回 `202 Accepted` 与 `{"task_id": "...", "status": "queued"}`，处理进度通过 `GET /v1/bills/{task_id}` 查询

#### Scenario: 提交 multipart 文件
- **WHEN** 客户端 POST `multipart/form-data`，`files` 字段含 1 个或多个 `UploadFile`
- **THEN** 接口为每个文件分配一个 `task_id`（`task_id = uuid`），按文件名排序后逐个入队，返回 202 + `{"task_ids": [...], "status": "queued"}`

#### Scenario: 提交图片 URL 列表
- **WHEN** 客户端 POST `application/json`，body 为 `{"image_urls": ["https://...", "file:///..."]}`
- **THEN** 服务端下载（用 SDK 现有的 `requests` 复用连接池）后入队，行为与 multipart 相同

#### Scenario: 提交空请求
- **WHEN** 请求体不包含 image / files / image_urls
- **THEN** 返回 400 + `{"error": "no image provided"}`

### Requirement: 异步结果查询接口

`GET /v1/bills/{task_id}` 必须返回三种状态之一：`queued` / `processing` / `done` / `failed`。

#### Scenario: 处理中
- **WHEN** 服务端还在解析
- **THEN** 返回 200 + `{"task_id": "...", "status": "processing", "progress": 0.4}`

#### Scenario: 处理完成
- **WHEN** PreprocessPool 回调完成、字段抽取完成
- **THEN** 返回 200 + `{"task_id": "...", "status": "done", "result": {BillRecord JSON}}`

#### Scenario: 失败
- **WHEN** 预处理崩溃、OCR 5xx 连续失败、或字段抽取 JSON 解析失败
- **THEN** 返回 200 + `{"task_id": "...", "status": "failed", "error": "..."}` （避免客户端按 5xx 重试；状态由 result 字段决定）

#### Scenario: task_id 不存在
- **WHEN** 查询未知 task_id
- **THEN** 返回 404 + `{"error": "task not found"}`

### Requirement: 健康检查

`GET /health` 必须 200。

#### Scenario: 服务存活
- **WHEN** 任何时刻
- **THEN** 返回 200 + `{"status": "ok"}`；如果 PreprocessPool 未启动则返回 503

### Requirement: 字段抽取

每个处理完成的票据，必须产出符合 `BillRecord` schema 的 JSON，字段包括：

```python
class BillItem(BaseModel):
    name: str          # 项目名称（如"CT 平扫"）
    quantity: float    # 数量
    unit_price: float  # 单价
    amount: float      # 金额

class BillRecord(BaseModel):
    bill_no: Optional[str]         # 票据编号
    hospital: Optional[str]        # 医院名称
    issue_date: Optional[str]      # 开票日期 (ISO 8601)
    total_amount: Optional[float]  # 总额
    self_pay: Optional[float]      # 自费
    insurance_pay: Optional[float] # 医保统筹
    items: List[BillItem]          # 项目明细
    raw_markdown: str              # 原始 OCR markdown
```

#### Scenario: 抽取成功
- **WHEN** LLM 返回的 JSON 包含所有字段
- **THEN** 服务端用 Pydantic 校验后写入 result；缺失字段保持 `None`、空 list 保持 `[]`

#### Scenario: 抽取 JSON 损坏
- **WHEN** LLM 返回的 JSON 解析失败
- **THEN** 自动重试 1 次（不同 temperature）；仍失败则 status=failed, error=LLM 输出前 200 字符

### Requirement: 任务生命周期

服务必须保留每个 `task_id` 的中间状态至少 1 小时（可配置），过期自动清理。

#### Scenario: 1 小时后查询已完成任务
- **WHEN** 当前时间 > 完成时间 + ttl_seconds
- **THEN** 返回 404，行为同"task_id 不存在"

#### Scenario: ttl=0
- **WHEN** 配置 `medical_server.task_ttl_seconds=0`
- **THEN** 表示永久保留（用于调试）

### Requirement: 并发模型

接口必须不阻塞 FastAPI 事件循环。

#### Scenario: 50 个并发提交
- **WHEN** 50 个客户端同时 POST
- **THEN** 服务端在 100ms 内全部返回 202；解析在后台 PreprocessPool 进程内并行（4 进程 × 32 OCR 线程）

#### Scenario: PreprocessPool 满载
- **WHEN** 队列内已有 200 个未完成任务
- **THEN** 新提交的请求仍可立刻入队（内存级 dict），但 join() 时间会变长；这不算 bug

## MODIFIED Requirements

无（这是全新模块，不修改任何既有功能）。

## REMOVED Requirements

无。

## Out of Scope

- 多用户鉴权 / 速率限制 / 配额
- 数据库持久化（用内存 dict + 进程内文件落盘可选）
- 任务取消 API
- WebSocket 推送（仅用 HTTP 轮询）
- 票据图像矫正：复用现有 `PreprocessPool` 的 4 阶段（用户负责实现 3 个检测器）
- 跨国多语言：仅中文医疗票据
