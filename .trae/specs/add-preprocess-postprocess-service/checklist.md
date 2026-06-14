# 预处理与后处理微服务检查清单

## 预处理服务
- [ ] `preprocess/` 目录已创建
- [ ] `preprocess/config.py` 已创建，包含预处理配置类
- [ ] `preprocess/models/` 目录已创建，包含模型加载器基类
- [ ] `preprocess/doc_detector.py` 已创建，实现文档边界检测
- [ ] `preprocess/orientation_corrector.py` 已创建，实现方向矫正
- [ ] `preprocess/dewarp_corrector.py` 已创建，实现扭曲矫正
- [ ] `preprocess/pipeline.py` 已创建，实现串行流水线
- [ ] `preprocess/server.py` 已创建，FastAPI 服务入口
- [ ] `/detect` 端点已实现
- [ ] `/orient` 端点已实现
- [ ] `/dewarp` 端点已实现
- [ ] `/preprocess` 端点已实现（完整流水线）
- [ ] `/health` 端点已实现
- [ ] `preprocess/config.yaml` 已创建
- [ ] `preprocess/Dockerfile` 已创建
- [ ] `preprocess/README.md` 已创建

## 后处理服务
- [ ] `postprocess/` 目录已创建
- [ ] `postprocess/processor.py` 已创建，实现后处理逻辑
- [ ] `postprocess/server.py` 已创建，FastAPI 服务入口
- [ ] `/postprocess` 端点已实现
- [ ] `/health` 端点已实现
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

## 测试
- [ ] 预处理服务单元测试已编写
- [ ] 后处理服务单元测试已编写
- [ ] 集成测试已编写
- [ ] 所有测试通过

## 文档
- [ ] 主 README 已更新，包含预处理/后处理说明
- [ ] API 文档已更新
- [ ] 部署文档已更新
