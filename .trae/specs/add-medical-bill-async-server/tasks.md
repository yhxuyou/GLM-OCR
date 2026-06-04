# Tasks

> 实现顺序：先 1，再 2，并行 3 & 4，最后 5。
> 每完成一项把 `- [ ]` 改成 `- [x]`。

- [ ] Task 1: 扩展 `glmocr/config.py` + `glmocr/config.yaml` 加入 `MedicalServerConfig` 与 `MedicalExtractorConfig`
  - [ ] SubTask 1.1: 在 `glmocr/config.py` 新增 `MedicalExtractorConfig` (field_definitions / prompt_template / retry_count / temperature)
  - [ ] SubTask 1.2: 在 `glmocr/config.py` 新增 `MedicalServerConfig` (host/port/max_concurrent_tasks/task_ttl_seconds/preprocessor_factory)
  - [ ] SubTask 1.3: 在 `GlmOcrConfig` 顶层加 `medical_server: MedicalServerConfig` 字段
  - [ ] SubTask 1.4: 在 `glmocr/config.yaml` 加对应 yaml 段（默认端口 8080、prompt 模板示例）

- [ ] Task 2: 新增 `glmocr/medical_extractor.py` —— 字段抽取器
  - [ ] SubTask 2.1: 定义 Pydantic 模型 `BillItem` / `BillRecord`
  - [ ] SubTask 2.2: 实现 `MedicalFieldExtractor`，用 `OCRClient.process` 调同一个 vLLM 后端，prompt 来自配置
  - [ ] SubTask 2.3: 实现 JSON 解析失败的重试（1 次不同 temperature）
  - [ ] SubTask 2.4: 加 `extract(markdown: str) -> BillRecord` 主入口

- [ ] Task 3: 新增 `glmocr/medical_server.py` —— FastAPI 服务
  - [ ] SubTask 3.1: `create_app(config, preprocessor_factory)` 工厂；启动时建 PreprocessPool + extractor
  - [ ] SubTask 3.2: 任务状态机（内存 dict + TTL 清理器后台协程）
  - [ ] SubTask 3.3: `POST /v1/bills/parse` —— 支持 base64 / multipart / URL
  - [ ] SubTask 3.4: `GET /v1/bills/{task_id}` —— 返回 status / result / error
  - [ ] SubTask 3.5: `GET /health` —— 返回 200/503
  - [ ] SubTask 3.6: 把 PreprocessPool 回调桥接到 asyncio.Future（`loop.call_soon_threadsafe`）
  - [ ] SubTask 3.7: `main()` CLI 入口（参考 `server.py` 的 argparse）

- [ ] Task 4: 写单测
  - [ ] SubTask 4.1: `glmocr/tests/test_medical_extractor.py` —— mock `OCRClient.process`，验证 BillRecord 解析、JSON 损坏重试
  - [ ] SubTask 4.2: `glmocr/tests/test_medical_server.py` —— 用 `httpx.AsyncClient + ASGITransport`（不依赖真实端口），验证 POST→GET 全流程、空请求 400、未知 task_id 404

- [ ] Task 5: 验证 + 文档
  - [ ] SubTask 5.1: 跑全量 `pytest glmocr/tests/ glmocr/preprocess_pool/tests/`，确认 0 回归
  - [ ] SubTask 5.2: 在 `glmocr/__init__.py` 暴露 `BillRecord` / `BillItem`（可选，懒加载）
  - [ ] SubTask 5.3: 用 curl 起 server，submit 1 张图 + GET 拉取，截图保存到本目录（可选）

# Task Dependencies

- Task 2 依赖 Task 1（要读配置里的 prompt）
- Task 3 依赖 Task 1（要读配置里的 host/port/ttl）+ Task 2（要 import extractor）
- Task 4.2 依赖 Task 3
- Task 5 依赖所有上面
