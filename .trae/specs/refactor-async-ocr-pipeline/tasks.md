# 异步 OCR Client 和 Pipeline 改造任务清单

## 已完成任务

- [x] 创建 AsyncOCRClient（httpx 异步客户端）
  - [x] 使用 httpx.AsyncClient 替代 requests
  - [x] 实现异步连接池管理
  - [x] 实现异步请求方法 async_process()
  - [x] 实现批量异步请求方法 async_process_batch()
  - [x] 实现异步重试逻辑
  - [x] 实现异步健康检查 is_alive()

- [x] 创建 AsyncPipeline（完全异步 Pipeline）
  - [x] 使用 AsyncOCRClient 替代 OCRClient
  - [x] 实现异步启动和停止方法
  - [x] 实现异步文档处理方法 process_async()
  - [x] 实现异步 region 提交 _submit_region_async()
  - [x] 使用 asyncio.Semaphore 控制并发
  - [x] 布局检测在后台线程执行（CPU 密集）
  - [x] OCR 提交完全异步

- [x] 更新配置（async_ocr 和 async_pipeline）
  - [x] 添加 AsyncOCRApiConfig 配置类
  - [x] 添加 AsyncPipelineConfig 配置类
  - [x] 在 PipelineConfig 中添加 async_ocr 和 async_pipeline 字段
  - [x] 更新 config.yaml 添加异步配置示例
  - [x] 添加环境变量支持

- [x] 更新 async_server.py 使用新组件
  - [x] 导入 AsyncPipeline 替代 Pipeline
  - [x] 在 startup_event 中使用 AsyncPipeline
  - [x] 在 shutdown_event 中异步停止 AsyncPipeline
  - [x] 更新后台处理函数使用 process_async()

- [x] 编写测试
  - [x] 创建 test_async_components.py
  - [x] 测试 AsyncOCRClient（6 个测试）
  - [x] 测试 AsyncPipeline（5 个测试）
  - [x] 测试配置类（3 个测试）
  - [x] 所有 14 个测试通过

- [x] 验证实现
  - [x] 验证模块导入成功
  - [x] 验证配置正确加载
  - [x] 验证 AsyncOCRClient 初始化
  - [x] 验证 AsyncPipeline 初始化（需要布局检测器依赖）

## 待完成任务

- [ ] 更新文档
  - [ ] 更新 README.md 添加异步服务说明
  - [ ] 创建异步服务使用示例
  - [ ] 添加 API 文档

- [ ] 性能测试
  - [ ] 对比同步和异步版本的性能
  - [ ] 测试高并发场景
  - [ ] 测试资源使用情况

## 任务依赖关系

无依赖关系，所有核心任务已完成。

## 备注

- AsyncPipeline 初始化需要布局检测器依赖（cv2），这是预期的环境依赖
- 所有单元测试已通过（14/14）
- 代码已实现 fire-and-forget 模式
- 向后兼容性保持良好
