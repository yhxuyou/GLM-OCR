# Checklist

> 验证 spec.md 里每条 Requirement 的可观察证据。
> 通过一项就把 `- [ ]` 改成 `- [x]`，并在末尾用一行注释引用证据（测试名 / 路径 / 行号）。

## Requirement: 同步入参校验
- [ ] 缺 `request_id` 返回 HTTP 400 + `{"code": "400", "message": "request_id is required", ...}` —— `glmocr/tests/test_medical_server.py::test_missing_request_id`
- [ ] 缺 `file_list` 返回 HTTP 400 + `"file_list is required and must be non-empty"` —— `glmocr/tests/test_medical_server.py::test_missing_file_list`
- [ ] `file_list` 元素缺 `page_count`/`file_type`/`file` 返回 HTTP 400 并指明哪个 file —— `glmocr/tests/test_medical_server.py::test_file_list_element_missing_field`
- [ ] `file_type` 非法返回 HTTP 400 + `"unsupported file_type: ..."` —— `glmocr/tests/test_file_fetcher.py::test_registry_unsupported_type`

## Requirement: image_id 生成规则
- [ ] `page_count=5` 的 file_id 产出 `id_1` ~ `id_5` 严格按 `{file_id}_{seq}` 格式 —— `glmocr/tests/test_medical_server.py::test_image_id_format`
- [ ] 多 file_id 时 seq 在每个 file_id 内独立从 1 开始 —— `glmocr/tests/test_medical_server.py::test_image_id_resets_per_file_id`
- [ ] 某张图下载失败时其 `image_id` 进入 `discarded_image`（含失败原因日志） —— `glmocr/tests/test_medical_server.py::test_partial_download_failure`

## Requirement: 异步内部流水线
- [ ] `alg_request_time` 字段在响应中 ≥ 0 且为 float —— `glmocr/tests/test_medical_server.py::test_alg_request_time_present`
- [ ] 30 张图提交后端到端时延 < 5s（mock 掉 OCR 实际推理） —— `glmocr/tests/test_medical_server.py::test_30_images_throughput`
- [ ] `/health` 在所有任务 in-flight 时仍 < 100ms 返回 —— `glmocr/tests/test_medical_server.py::test_health_during_processing`
- [ ] 内部用了 `PreprocessPool`（验证：mock PreprocessPool，断言 submit 被调用） —— `glmocr/tests/test_medical_server.py::test_uses_preprocess_pool`

## Requirement: 字段抽取（BillRecord）
- [ ] `BillRecord` schema 包含 image_id / bill_no / hospital / issue_date / total_amount / self_pay / insurance_pay / items / raw_markdown —— `glmocr/tests/test_medical_extractor.py::test_bill_record_schema`
- [ ] 缺失字段保持 `None`，空 list 保持 `[]` —— `glmocr/tests/test_medical_extractor.py::test_missing_fields_default`
- [ ] JSON 损坏时自动重试 1 次不同 temperature —— `glmocr/tests/test_medical_extractor.py::test_json_decode_retry`
- [ ] 仍失败时该图计入 `discarded_image`（而非整体失败） —— `glmocr/tests/test_medical_server.py::test_extraction_failure_keeps_other_images`
- [ ] 整张图 OCR 失败时该 image_id 进入 `discarded_image` —— `glmocr/tests/test_medical_server.py::test_ocr_failure_discards_image`

## Requirement: 票据-病历匹配与 case 维度聚合
- [ ] 票据 (hospital, issue_date) 与病历 (hospital_name, ±N 天) 命中后归入对应 `case_id` —— `glmocr/tests/test_medical_aggregator.py::test_match_by_hospital_and_date`
- [ ] 同 case 多张票据求和为 `bill_total` —— `glmocr/tests/test_medical_aggregator.py::test_bill_total_sum`
- [ ] 病历列表为空时所有票据进 `discarded_image`，`case_info == []` —— `glmocr/tests/test_medical_aggregator.py::test_no_medical_records`
- [ ] OCR 内容 hash 一致的图只保留首张 —— `glmocr/tests/test_medical_aggregator.py::test_dedup_duplicate_markdown`
- [ ] 未匹配上的票据 image_id 进入 `discarded_image`，`case_info` 不出现 —— `glmocr/tests/test_medical_aggregator.py::test_unmatched_bill_discarded`

## Requirement: discarded_image 规则
- [ ] discarded_image 列表内 seq 升序 —— `glmocr/tests/test_medical_server.py::test_discarded_image_sorted`
- [ ] 覆盖所有失败场景（下载/OCR/抽取/未匹配/重复） —— `glmocr/tests/test_medical_server.py::test_all_discard_paths`

## Requirement: 健康检查
- [ ] `GET /health` 在 PreprocessPool 已启动时返回 200 + `{"status": "ok"}` —— `glmocr/tests/test_medical_server.py::test_health_ok`
- [ ] `GET /health` 在 PreprocessPool 未启动时返回 503 —— `glmocr/tests/test_medical_server.py::test_health_pool_not_started`

## Requirement: 配置与启动
- [ ] `python -m glmocr.medical_server --help` 输出帮助（包含 `--config` / `--port` / `--host`） —— `glmocr/tests/test_medical_server.py::test_cli_help`
- [ ] `--port` / `--config` 覆盖 yaml 默认值 —— `glmocr/tests/test_medical_server.py::test_cli_overrides`

## 通用
- [ ] `pytest glmocr/tests/test_file_fetcher.py` 全绿
- [ ] `pytest glmocr/tests/test_medical_extractor.py` 全绿
- [ ] `pytest glmocr/tests/test_medical_aggregator.py` 全绿
- [ ] `pytest glmocr/tests/test_medical_server.py` 全绿
- [ ] 现有 174 个测试无回归（`pytest glmocr/preprocess_pool/tests/`）
- [ ] curl 端到端：提交真实样例入参 → 校验出参含 `code=200`、`discarded_image[]` 顺序、`case_info[].bills` 长度
