# 医疗票据异步解析服务 Spec（PA_OCR 接入版）

## Why

保险公司理赔系统（PA_OCR）需要把一次理赔案件下挂的多张医疗票据图片（住院/门诊/检查/药品等）一次性送进 OCR 引擎做结构化抽取，再按"住院维度"聚合字段喂给后端核赔。之前的 spec 把接口设计成"提交 → 拿 task_id → 轮询"的纯异步模型，但实际生产里 PA_OCR 调用方期望**一次 HTTP 调用拿到完整结果**（带回 `request_id`、`alg_request_time`、`discarded_image`、`case_info`），便于上游同步落库。

为了让内部能扛住单案件 5~30 张图、并发 50+ 案件的流量，我们继续复用已经搭好的 `PreprocessPool`（4 进程预处理 + 异步 OCR + 按文档聚合），把"同步入参/出参"和"内部异步流水线"解耦。本 spec 同时确定字段抽取 schema 与"按病历 → 住院 case 维度聚合"的归并规则。

## What Changes

- **新增** `glmocr/medical_extractor.py` — 医疗票据字段抽取器，把 OCR 出的 markdown 送给同一个 vLLM 后端做结构化抽取，输出 Pydantic `BillRecord`。
- **新增** `glmocr/medical_server.py` — FastAPI 同步响应服务（`POST /v1/bills/parse`），内部用 `PreprocessPool` 并行处理，HTTP 200 一次性返回完整结果。
- **新增** `glmocr/medical_aggregator.py` — 病历-票据匹配 + 住院 case 维度归并（`discarded_image` / `case_info` 生成）。
- **新增** `glmocr/file_fetcher.py` — 根据 `file_type` 选择不同的文件获取策略（`file_id` 默认走 HTTP 远程拉取，可扩展 oss/s3）。
- **新增** `glmocr/config.py` 的 `MedicalServerConfig` / `MedicalExtractorConfig` / `FileFetcherConfig` 配置块；`GlmOcrConfig` 增加 `medical_server` 字段。
- **新增** `glmocr/config.yaml` 对应配置段。
- **新增** 单测覆盖：字段抽取、HTTP 路由、文件拉取、聚合归并。
- **不修改** `glmocr/server.py`（保留向后兼容）。
- **不修改** `PreprocessPool` 本身（仅作为客户端调用）。

## Impact

- **Affected specs**：无既有 spec；这是新能力。
- **Affected code**：
  - 新增：`glmocr/medical_extractor.py`、`glmocr/medical_server.py`、`glmocr/medical_aggregator.py`、`glmocr/file_fetcher.py`
  - 扩展：`glmocr/config.py`、`glmocr/config.yaml`、`glmocr/__init__.py`（懒加载导出 `BillRecord`）
  - 测试：`glmocr/tests/test_medical_extractor.py`、`glmocr/tests/test_medical_server.py`、`glmocr/tests/test_medical_aggregator.py`、`glmocr/tests/test_file_fetcher.py`
- **新依赖**：`fastapi`、`uvicorn`、`httpx`（test client / 远程拉取图片）；其余复用 `PreprocessPool` 与 `OCRClient`。
- **运行方式**：`python -m glmocr.medical_server --config glmocr/config.yaml --port 8080`。

## 入参 / 出参契约

### 入参

```json
{
  "request_id": "2c2c1430981a11eb98aea4c3f0f414c3",
  "system": "PA_OCR",
  "regsno": "MC010000",
  "file_list": [
    {
      "page_count": 5,
      "file_type": "file_id",
      "file": "906ac094ad28d554ab2b974bfc738c4b9cd"
    }
  ],
  "pass_through_data": {
    "medical_records": [
      {
        "medical_id": "001",
        "hospital_name": "深圳市第一人民医院",
        "outpatientDate": "2022-08-26",
        "startDate": "2022-08-26",
        "endDate": "2022-08-26"
      },
      {
        "medical_id": "002",
        "hospital_name": "南山协和人民医院",
        "outpatientDate": "2022-08-26",
        "startDate": "2022-08-26",
        "endDate": "2022-08-26"
      }
    ]
  }
}
```

字段语义：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `request_id` | 是 | 调用方请求唯一 ID，长度 32 字符（UUID 去连字符）。出参原样回传。 |
| `system` | 否 | 调用方系统名，默认 `"PA_OCR"`。 |
| `regsno` | 是 | 报案号/案件号，出参原样回传。 |
| `file_list` | 是 | 文件列表，每个元素代表一个 case 内的一种"影像"（file_id）下挂多张图片。 |
| `file_list[].page_count` | 是 | 该 file_id 对应图片总张数；引擎下载后按 **1 开始编号** 拼接 `file_id` 作为出参的 `image_id`。 |
| `file_list[].file_type` | 是 | 数据源类型；当前支持 `"file_id"`（通过 `file_fetcher` 远程拉取），保留扩展位（`"url"`/`"oss"`/`"base64"` 后续可加）。 |
| `file_list[].file` | 是 | 当 `file_type="file_id"` 时为 `file_id` 字符串（图片托管系统会按规则拼 URL）。 |
| `pass_through_data.medical_records` | 否 | 上游已抽取的病历元信息（医院+日期），用于把 OCR 出来的票据**匹配**到对应住院 case。 |
| `medical_records[].medical_id` | 是 | 病历 ID（如 `"001"`），出参原样回传。 |
| `medical_records[].hospital_name` | 否 | 医院名称；用于票据-病历匹配。 |
| `medical_records[].outpatientDate` | 否 | 门诊日期（`YYYY-MM-DD`），空字符串表示缺省。 |
| `medical_records[].startDate` | 否 | 入院日期。 |
| `medical_records[].endDate` | 否 | 出院日期。 |

### 出参

```json
{
  "request_id": "2c2c1430981a11eb98aea4c3f0f414c3",
  "code": "200",
  "message": "success",
  "alg_request_time": 4.1720287799835205,
  "regsno": "MC010000",
  "pass_through_data": {
    "medical_records": [
      { "medical_id": "001", "hospital_name": "...", "outpatientDate": "...", "startDate": "...", "endDate": "..." },
      { "medical_id": "002", "hospital_name": "...", "outpatientDate": "...", "startDate": "...", "endDate": "..." }
    ]
  },
  "discarded_image": [
    "906ac094ad28d554ab2b974bfc738c4b9cd_3",
    "906ac094ad28d554ab2b974bfc738c4b9cd_4"
  ],
  "case_info": [
    {
      "case_id": "001",
      "hospital_name": "深圳市第一人民医院",
      "bill_total": 1234.56,
      "bills": [
        { "image_id": "906ac094ad28d554ab2b974bfc738c4b9cd_1", "bill_no": "...", "items": [...] }
      ]
    }
  ]
}
```

字段语义：

| 字段 | 必返 | 说明 |
| --- | --- | --- |
| `request_id` | 是 | 原样回传。 |
| `code` | 是 | `"200"`=业务成功；`"400"`=参数错误；`"500"`=系统异常。 |
| `message` | 是 | 人类可读说明，`success`/`no image provided`/`internal error` 等。 |
| `alg_request_time` | 是 | 算法处理总耗时（秒，float），从收到请求到 `discarded_image`+`case_info` 全部产出为止。 |
| `regsno` | 是 | 原样回传。 |
| `pass_through_data.medical_records` | 是 | 原样回传（不修改）。 |
| `discarded_image` | 是 | 未纳入信息整合的 `image_id` 列表，可能为重复图、非清单图、异常图（OCR 失败/图片解码失败）。`image_id` 格式严格为 `{file_id}_{seq}`，seq 从 1 开始。 |
| `case_info` | 是 | 按"住院/就诊 case 维度"归并的结构化结果；每个 `case` 至少包含 `case_id`(=`medical_id`)、`hospital_name`、`bill_total`（该 case 下所有票据金额求和）、`bills[]`（该 case 下所有票据明细）。 |

> 同步返回的 HTTP 状态码：业务成功 → 200；参数错误 → 400；系统异常 → 500。**HTTP 200 时 `code` 字段才是唯一权威**，避免与 `request_id` 状态码混淆。

## ADDED Requirements

### Requirement: 同步入参校验

服务必须严格按入参 schema 校验输入，缺关键字段立刻拒绝。

#### Scenario: 缺少 request_id
- **WHEN** 请求体不含 `request_id` 或为空字符串
- **THEN** 返回 HTTP 400 + body `{"code": "400", "message": "request_id is required", "request_id": ""}`

#### Scenario: 缺少 file_list
- **WHEN** 请求体不含 `file_list` 或为空数组
- **THEN** 返回 HTTP 400 + body `{"code": "400", "message": "file_list is required and must be non-empty"}`

#### Scenario: file_list 元素缺字段
- **WHEN** 某 `file_list[i]` 缺 `page_count`/`file_type`/`file`
- **THEN** 返回 HTTP 400 + body 指明哪个 `file` 出错

#### Scenario: file_type 不支持
- **WHEN** `file_type` 不是 `"file_id"`（或任何当前已实现的类型）
- **THEN** 返回 HTTP 400 + body `"unsupported file_type: <value>"`

### Requirement: image_id 生成规则

引擎必须为每张成功下载的图片生成稳定的 `image_id`，格式严格为 `{file_id}_{seq}`，seq 从 1 开始按 `file_list` 内顺序递增。

#### Scenario: 5 页 file_id
- **WHEN** `file_list[0] = {"file": "abc", "page_count": 5, "file_type": "file_id"}`
- **THEN** 生成 5 个 `image_id`：`abc_1` ~ `abc_5`

#### Scenario: 多 file_id 跨案件
- **WHEN** `file_list` 包含 2 个 `file_id`，分别 3 张和 2 张
- **THEN** 第一个 `file_id` 产出 `id1_1..id1_3`，第二个产出 `id2_1..id2_2`；seq 在每个 `file_id` 内独立从 1 开始

#### Scenario: 某张图下载失败
- **WHEN** `file_id` 声明 5 张但实际只能下到 3 张
- **THEN** 成功下载的 3 张正常出 `image_id` 并进入处理；下载失败的 2 张 `image_id` 直接进入 `discarded_image`，错误原因记入日志

### Requirement: 异步内部流水线

服务对外同步返回，但**内部**必须用 `PreprocessPool` 并行处理多张图，确保单案件 5~30 张图的端到端时延 ≤ 5s（受 OCR 推理速度影响可放宽，由 `alg_request_time` 字段观测）。

#### Scenario: 30 张图并发处理
- **WHEN** 单案件 30 张图
- **THEN** 4 个预处理进程 + 32 个 OCR 线程并发；`alg_request_time` 应显著小于串行处理时延（基线由 `tests/perf` 校准）

#### Scenario: 50 案件并发请求
- **WHEN** 50 个不同 `request_id` 的请求同时打进来
- **THEN** 每个请求独立调度到 `PreprocessPool`；FastAPI worker 数 ≥ 4，事件循环不阻塞

### Requirement: 字段抽取（BillRecord）

每张成功 OCR 的票据必须产出符合 `BillRecord` schema 的 JSON：

```python
class BillItem(BaseModel):
    name: str          # 项目名称
    quantity: float    # 数量
    unit_price: float  # 单价
    amount: float      # 金额

class BillRecord(BaseModel):
    image_id: str                    # 关联的 image_id
    bill_no: Optional[str]           # 票据编号
    hospital: Optional[str]          # 医院名称
    issue_date: Optional[str]        # 开票日期 (ISO 8601)
    total_amount: Optional[float]    # 总额
    self_pay: Optional[float]        # 自费
    insurance_pay: Optional[float]   # 医保统筹
    items: List[BillItem]            # 项目明细
    raw_markdown: str                # 原始 OCR markdown（便于溯源）
```

#### Scenario: 抽取成功
- **WHEN** LLM 返回的 JSON 包含所有字段
- **THEN** 服务端用 Pydantic 校验后写入 result；缺失字段保持 `None`、空 list 保持 `[]`

#### Scenario: 抽取 JSON 损坏
- **WHEN** LLM 返回的 JSON 解析失败
- **THEN** 自动重试 1 次（不同 temperature）；仍失败则该图计入 `discarded_image`，错误记入日志

#### Scenario: 整张图 OCR 失败
- **WHEN** PreprocessPool 上报图片解码失败 / 布局检测无 region / OCR 连续 5xx
- **THEN** 该 `image_id` 进入 `discarded_image`，不阻塞其他图片

### Requirement: 票据-病历匹配与 case 维度聚合

`medical_aggregator` 必须把 OCR 出来的每张 `BillRecord` 关联到 `pass_through_data.medical_records` 中的某条病历，再按 `case_id`(=`medical_id`) 归并成 `case_info`。

#### Scenario: 匹配规则
- **WHEN** 票据 OCR 出 `hospital` 字段
- **THEN** 用 `(hospital, ±N 天 issue_date)` 与病历 `(hospital_name, startDate/endDate)` 做模糊匹配；命中即归入对应 `case_id`；未命中归入 `case_id="unmatched"`（不出现于 `case_info`，其 `image_id` 进入 `discarded_image`）

#### Scenario: 同 case 多张票据
- **WHEN** case_id="001" 下命中 3 张票据
- **THEN** `case_info[i].bills` 长度为 3，`case_info[i].bill_total = sum(bill.total_amount)`

#### Scenario: 病历列表为空
- **WHEN** `pass_through_data.medical_records` 为空或未传
- **THEN** 所有票据全部进入 `discarded_image`（无法归并），`case_info` 为 `[]`

#### Scenario: 重复图去重
- **WHEN** 多张图 OCR 内容 hash 相同
- **THEN** 保留第一张，后续重复 `image_id` 进入 `discarded_image`（reason="duplicate"）

### Requirement: discarded_image 规则

`discarded_image` 列出所有"未参与 case 归并"的 `image_id`。

#### Scenario: 出现场景
- 重复图（hash 一致）
- 票据-病历未匹配上
- 图片下载失败
- OCR 全流程失败（图片解码 / 布局检测 / OCR / 字段抽取）

#### Scenario: 顺序保证
- **WHEN** 同一 file_id 内出现多个 `discarded_image`
- **THEN** seq 升序输出（`id_1, id_2, ...`），便于上游对账

### Requirement: 健康检查

`GET /health` 必须 200。

#### Scenario: 服务存活
- **WHEN** 任何时刻
- **THEN** 返回 200 + `{"status": "ok"}`；如果 PreprocessPool 未启动则返回 503

### Requirement: 配置与启动

服务必须能从 `config.yaml` 加载所有配置项，并提供 CLI 入口。

#### Scenario: 默认启动
- **WHEN** `python -m glmocr.medical_server`
- **THEN** 监听 `0.0.0.0:8080`（来自 `medical_server.host`/`port`）

#### Scenario: 显式指定
- **WHEN** `python -m glmocr.medical_server --port 9000 --config /path/to/config.yaml`
- **THEN** 用指定端口与配置启动；`--port` 覆盖 yaml，`--config` 覆盖默认路径

## MODIFIED Requirements

无（这是全新模块，不修改任何既有功能）。

## REMOVED Requirements

- 移除上一版 spec 中的"基于 `task_id` 的异步轮询接口"（`POST /v1/bills/parse` + `GET /v1/bills/{task_id}`）。新 spec 改为**一次调用同步返回完整结果**。
- 移除上一版 spec 中的"base64 / multipart / URL 三种提交方式"。本版固定为 `file_id` 远程拉取，提交体只接受 JSON。

## Out of Scope

- 多用户鉴权 / 速率限制 / 配额
- 数据库持久化（结果只在内存）
- WebSocket 推送
- 票据图像矫正的具体实现：复用现有 `PreprocessPool` 的 4 阶段
- 跨国多语言：仅中文医疗票据
- `file_type` 扩展（`url`/`oss`/`base64`）：保留 `FileFetcher` 协议位，后续 PR 增量支持
- 增量/流式响应：单次请求 5~30 张图同步返回已够用
