# Tasks

> 实现顺序：先 1，并行 2/3/4/5，再 6，最后 7。
> 每完成一项把 `- [ ]` 改成 `- [x]`。

- [ ] Task 1: 扩展 `glmocr/config.py` + `glmocr/config.yaml` 加入三块新配置
  - [ ] SubTask 1.1: 在 `glmocr/config.py` 新增 `FileFetcherConfig`（`file_id_base_url` / `timeout` / `retries` / `headers`）
  - [ ] SubTask 1.2: 在 `glmocr/config.py` 新增 `MedicalExtractorConfig`（`prompt_template` / `retry_count` / `temperature` / `match_date_window_days`）
  - [ ] SubTask 1.3: 在 `glmocr/config.py` 新增 `MedicalServerConfig`（`host` / `port` / `max_concurrent_tasks` / `preprocessor_factory` / `supported_file_types`）
  - [ ] SubTask 1.4: 在 `GlmOcrConfig` 顶层加 `medical_server: MedicalServerConfig` 字段
  - [ ] SubTask 1.5: 在 `glmocr/config.yaml` 加对应 yaml 段（默认端口 8080、`file_id_base_url` 占位、prompt 模板示例）

- [ ] Task 2: 新增 `glmocr/file_fetcher.py` —— 文件拉取器
  - [ ] SubTask 2.1: 定义 `FileFetcher` 协议类（`fetch(file_id: str, seq: int) -> bytes`） + `FileFetchError`
  - [ ] SubTask 2.2: 实现 `FileIdFetcher`，根据 `file_id_base_url` + `{file_id}_{seq}` 拼 URL，HTTP GET 拿图片字节；超时/404/5xx 抛 `FileFetchError`
  - [ ] SubTask 2.3: 实现 `FileFetcherRegistry`，根据 `file_type` 字符串派发到具体 fetcher；未注册类型抛 `UnsupportedFileType`

- [ ] Task 3: 新增 `glmocr/medical_extractor.py` —— 字段抽取器
  - [ ] SubTask 3.1: 定义 Pydantic 模型 `BillItem` / `BillRecord`（含 `image_id`）
  - [ ] SubTask 3.2: 实现 `MedicalFieldExtractor`，用 `OCRClient.process` 调同一个 vLLM 后端，prompt 来自配置
  - [ ] SubTask 3.3: 实现 JSON 解析失败的重试（1 次不同 temperature）；仍失败抛 `FieldExtractionError`
  - [ ] SubTask 3.4: 加 `extract(image_id, markdown) -> BillRecord` 主入口

- [ ] Task 4: 新增 `glmocr/medical_aggregator.py` —— 病历-票据匹配 + case 归并
  - [ ] SubTask 4.1: 定义 `MedicalRecord` / `CaseInfo` / `AggregationResult` Pydantic 模型
  - [ ] SubTask 4.2: 实现 `_match_bill_to_medical(bill, medical_records, date_window_days) -> Optional[medical_id]`
  - [ ] SubTask 4.3: 实现 `MedicalAggregator.aggregate(bills, medical_records) -> AggregationResult`，输出 `discarded_image[]` + `case_info[]`
  - [ ] SubTask 4.4: 重复图去重：同 `image_id` 列表内 hash 一致只保留首张；`raw_markdown` hash 用 SHA1 前 16 字节

- [ ] Task 5: 新增 `glmocr/medical_server.py` —— FastAPI 服务
  - [ ] SubTask 5.1: `create_app(config, preprocessor_factory)` 工厂；启动时建 PreprocessPool + extractor + aggregator + FileFetcherRegistry
  - [ ] SubTask 5.2: `ParseRequest` / `ParseResponse` Pydantic 模型（含 `request_id` / `code` / `message` / `alg_request_time` / `regsno` / `pass_through_data` / `discarded_image` / `case_info`）
  - [ ] SubTask 5.3: `POST /v1/bills/parse` 路由：入参校验 → 拉取图片 → 并行送入 PreprocessPool + extractor → 调 aggregator → 组装出参
  - [ ] SubTask 5.4: `GET /health` 路由：200/503
  - [ ] SubTask 5.5: 把 PreprocessPool 回调桥接到 asyncio.Future（`loop.call_soon_threadsafe`）；计算 `alg_request_time`
  - [ ] SubTask 5.6: 全局异常处理：未捕获异常 → HTTP 500 + `{"code": "500", "message": "internal error", "request_id": "<原值>"}`
  - [ ] SubTask 5.7: `main()` CLI 入口（参考 `server.py` 的 argparse，支持 `--config` / `--port` / `--host`）

- [ ] Task 6: 写单测
  - [ ] SubTask 6.1: `glmocr/tests/test_file_fetcher.py` —— mock `httpx`，验证 URL 拼装、超时重试、`UnsupportedFileType`
  - [ ] SubTask 6.2: `glmocr/tests/test_medical_extractor.py` —— mock `OCRClient.process`，验证 `BillRecord` 解析、JSON 损坏重试、`image_id` 透传
  - [ ] SubTask 6.3: `glmocr/tests/test_medical_aggregator.py` —— 验证匹配规则（命中/未命中/重复图/空病历）、`bill_total` 求和
  - [ ] SubTask 6.4: `glmocr/tests/test_medical_server.py` —— 用 `httpx.AsyncClient + ASGITransport`（不依赖真实端口），验证 POST 全流程：成功/400 校验/500 异常/缺字段；`alg_request_time > 0`；`discarded_image` 顺序

- [ ] Task 7: 验证 + 文档
  - [ ] SubTask 7.1: 跑全量 `pytest glmocr/tests/ glmocr/preprocess_pool/tests/`，确认 0 回归
  - [ ] SubTask 7.2: 在 `glmocr/__init__.py` 暴露 `BillRecord` / `BillItem` / `MedicalAggregator`（可选，懒加载）
  - [ ] SubTask 7.3: 用 curl 起 server，提交一份真实入参样例 + 校验出参 schema

# Task Dependencies

- Task 3 依赖 Task 1（要读配置里的 prompt / temperature）
- Task 4 依赖 Task 3（消费 `BillRecord`）
- Task 5 依赖 Task 1 + Task 2 + Task 3 + Task 4
- Task 6 各项独立，仅 6.4 依赖 Task 5
- Task 7 依赖所有上面
