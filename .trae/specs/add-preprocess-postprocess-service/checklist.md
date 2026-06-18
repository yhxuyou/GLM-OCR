# 预处理与后处理微服务检查清单

## 预处理服务
- [x] `preprocess/` 目录已创建
- [x] `preprocess/config.py` 已创建，包含预处理配置类
- [x] `preprocess/models/` 目录已创建，包含模型加载器基类
- [x] `preprocess/doc_detector.py` 已创建，实现文档边界检测
- [x] `preprocess/orientation_corrector.py` 已创建，实现方向矫正
- [x] `preprocess/dewarp_corrector.py` 已创建，实现扭曲矫正
- [x] `preprocess/pipeline.py` 已创建，实现串行流水线
- [x] `preprocess/server.py` 已创建，FastAPI 服务入口
- [x] `/detect` 端点已实现
- [x] `/orient` 端点已实现
- [x] `/dewarp` 端点已实现
- [x] `/preprocess` 端点已实现（完整流水线）
- [x] `/health` 端点已实现
- [x] `preprocess/config.yaml` 已创建
- [ ] `preprocess/Dockerfile` 已创建
- [ ] `preprocess/README.md` 已创建

## 后处理服务
- [x] `postprocess/` 目录已创建
- [x] `postprocess/processor.py` 已创建，实现后处理逻辑
- [x] `postprocess/server.py` 已创建，FastAPI 服务入口
- [x] `/postprocess` 端点已实现
- [x] `/health` 端点已实现
- [ ] `postprocess/config.yaml` 已创建
- [ ] `postprocess/Dockerfile` 已创建
- [ ] `postprocess/README.md` 已创建

## 客户端集成
- [ ] `glmocr/preprocess_client.py` 已创建
- [ ] `glmocr/postprocess_client.py` 已创建
- [ ] 预处理客户端支持单步骤调用
- [ ] 预处理客户端支持完整流水线调用
- [ ] 后处理客户端支持结果处理调用

## Pipeline 集成
- [ ] `glmocr/async_pipeline.py` 已更新，支持预处理步骤
- [ ] `glmocr/async_pipeline.py` 已更新，支持后处理步骤
- [ ] 配置类已更新，支持预处理/后处理配置
- [ ] `glmocr/config.yaml` 已更新，包含预处理/后处理配置段

## Docker 部署
- [ ] `docker-compose.preprocess.yml` 已创建（独立部署）
- [ ] `docker-compose.full.yml` 已创建（完整部署）

## Supervisor 进程管理
- [x] `supervisor/supervisord.conf` 已创建（Supervisor 主配置）
- [x] `supervisor/conf.d/preprocess.conf` 已创建（预处理服务配置）
- [x] `supervisor/conf.d/glmocr-async.conf` 已创建（GLM OCR 服务配置）
- [x] `supervisor/conf.d/postprocess.conf` 已创建（后处理服务配置）
- [x] `supervisor/generate_configs.py` 已创建（配置生成器）
- [x] `supervisor/manage.sh` 已创建（服务管理脚本）
- [x] 支持灵活调整各服务进程数量
- [x] 支持动态扩缩容

## 测试
- [ ] 预处理服务单元测试已编写
- [ ] 后处理服务单元测试已编写
- [ ] 集成测试已编写
- [ ] 所有测试通过

## 文档
- [ ] 主 README 已更新，包含预处理/后处理说明
- [ ] API 文档已更新
- [ ] 部署文档已更新
