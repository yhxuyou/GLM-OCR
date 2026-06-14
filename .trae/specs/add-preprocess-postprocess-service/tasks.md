# 预处理与后处理微服务任务清单

## Task 1: 创建预处理服务基础结构
- [x] 1.1 创建 `preprocess/` 目录结构
- [x] 1.2 创建 `preprocess/config.py` - 预处理配置类
- [x] 1.3 创建 `preprocess/models/` - 模型加载器基类

## Task 2: 实现文档检测服务
- [x] 2.1 创建 `preprocess/doc_detector.py` - 文档边界检测
- [x] 2.2 实现 `/detect` 端点
- [x] 2.3 实现模型加载和推理逻辑

## Task 3: 实现方向矫正服务
- [x] 3.1 创建 `preprocess/orientation_corrector.py` - 方向矫正
- [x] 3.2 实现 `/orient` 端点
- [x] 3.3 实现角度预测和旋转逻辑

## Task 4: 实现扭曲矫正服务
- [x] 4.1 创建 `preprocess/dewarp_corrector.py` - 扭曲矫正
- [x] 4.2 实现 `/dewarp` 端点
- [x] 4.3 实现扭曲矫正推理逻辑

## Task 5: 实现串行预处理流水线
- [x] 5.1 创建 `preprocess/pipeline.py` - 串行流水线
- [x] 5.2 实现 `/preprocess` 端点（完整流水线）
- [x] 5.3 实现步骤间的数据传递

## Task 6: 创建预处理服务入口
- [ ] 6.1 创建 `preprocess/server.py` - FastAPI 服务
- [ ] 6.2 实现健康检查端点
- [ ] 6.3 实现服务启动脚本

## Task 7: 创建后处理服务
- [ ] 7.1 创建 `postprocess/` 目录结构
- [ ] 7.2 创建 `postprocess/processor.py` - 后处理逻辑
- [ ] 7.3 创建 `postprocess/server.py` - FastAPI 服务
- [ ] 7.4 实现 `/postprocess` 端点

## Task 8: 创建预处理客户端
- [ ] 8.1 创建 `glmocr/preprocess_client.py`
- [ ] 8.2 实现异步 HTTP 客户端
- [ ] 8.3 实现单步骤和完整流水线调用

## Task 9: 创建后处理客户端
- [ ] 9.1 创建 `glmocr/postprocess_client.py`
- [ ] 9.2 实现异步 HTTP 客户端
- [ ] 9.3 实现后处理调用逻辑

## Task 10: 集成到 AsyncPipeline
- [ ] 10.1 修改 `glmocr/async_pipeline.py` - 添加预处理步骤
- [ ] 10.2 修改 `glmocr/async_pipeline.py` - 添加后处理步骤
- [ ] 10.3 更新配置类支持预处理/后处理配置

## Task 11: 创建配置文件
- [ ] 11.1 创建 `preprocess/config.yaml` - 预处理配置示例
- [ ] 11.2 创建 `postprocess/config.yaml` - 后处理配置示例
- [ ] 11.3 更新 `glmocr/config.yaml` - 添加预处理/后处理配置段

## Task 12: 创建进程管理配置
- [x] 12.1 创建 `supervisor/supervisord.conf` - Supervisor 主配置
- [x] 12.2 创建 `supervisor/conf.d/preprocess.conf` - 预处理服务配置
- [x] 12.3 创建 `supervisor/conf.d/glmocr-async.conf` - GLM OCR 服务配置
- [x] 12.4 创建 `supervisor/conf.d/postprocess.conf` - 后处理服务配置
- [x] 12.5 创建 `supervisor/generate_configs.py` - 配置生成器
- [x] 12.6 创建 `supervisor/manage.sh` - 服务管理脚本

## Task 13: 编写测试
- [ ] 13.1 编写预处理服务单元测试
- [ ] 13.2 编写后处理服务单元测试
- [ ] 13.3 编写集成测试

## Task 14: 编写文档
- [ ] 14.1 创建 `preprocess/README.md` - 预处理服务文档
- [ ] 14.2 创建 `postprocess/README.md` - 后处理服务文档
- [ ] 14.3 更新主 README - 添加预处理/后处理说明

## 任务依赖关系
- Task 2, 3, 4 依赖 Task 1
- Task 5 依赖 Task 2, 3, 4
- Task 6 依赖 Task 5
- Task 7 独立
- Task 8 依赖 Task 6
- Task 9 依赖 Task 7
- Task 10 依赖 Task 8, 9
- Task 11 依赖 Task 10
- Task 12 依赖 Task 6, 7
- Task 13 依赖 Task 10
- Task 14 依赖 Task 13
