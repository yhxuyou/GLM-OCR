# Checklist

> 验证 spec.md 里每条 Requirement 的可观察证据。
> 通过一项就把 `- [ ]` 改成 `- [x]`，并在末尾用一行注释引用证据（测试名 / 路径 / 行号）。

## Requirement: 异步提交接口
- [ ] `POST /v1/bills/parse` 接受 base64 图片，立即返回 202 + task_id
- [ ] `POST /v1/bills/parse` 接受 multipart 多文件，逐个分配 task_id，按文件名排序
- [ ] `POST /v1/bills/parse` 接受 image_urls 列表，下载后入队
- [ ] 空请求返回 400 + `{"error": "no image provided"}`

## Requirement: 异步结果查询接口
- [ ] 处理中返回 `{"status": "processing", "progress": 0.4}`
- [ ] 处理完成返回 `{"status": "done", "result": BillRecord}`
- [ ] 失败返回 `{"status": "failed", "error": "..."}`，HTTP 仍是 200
- [ ] 未知 task_id 返回 404 + `{"error": "task not found"}`

## Requirement: 健康检查
- [ ] `GET /health` 在 PreprocessPool 已启动时返回 200
- [ ] `GET /health` 在 PreprocessPool 未启动时返回 503

## Requirement: 字段抽取
- [ ] `BillRecord` schema 包含 bill_no / hospital / issue_date / total_amount / self_pay / insurance_pay / items / raw_markdown
- [ ] 缺失字段保持 `None`，空 list 保持 `[]`
- [ ] JSON 损坏时自动重试 1 次不同 temperature
- [ ] 仍失败时 status=failed 且 error 含 LLM 输出前 200 字符

## Requirement: 任务生命周期
- [ ] 已完成任务在 ttl_seconds 后查询返回 404
- [ ] ttl=0 表示永久保留

## Requirement: 并发模型
- [ ] 50 个并发提交在 100ms 内全部返回 202（受 CI 机器影响可放宽到 500ms）
- [ ] 接口不阻塞事件循环（验证：`/health` 在所有任务 in-flight 时仍 < 50ms 返回）

## 通用
- [ ] `pytest glmocr/tests/test_medical_extractor.py` 全绿
- [ ] `pytest glmocr/tests/test_medical_server.py` 全绿
- [ ] 现有 174 个测试无回归
- [ ] `python -m glmocr.medical_server --help` 输出帮助
