# 端口复用实现计划：Uvicorn Workers 方案

## 背景

当前 Supervisor 配置中，每个服务启动多个进程（numprocs=2 或 4），每个进程都尝试绑定同一个端口，导致端口冲突。

用户希望实现端口复用：每个服务对外只暴露一个端口，内部多进程共享这个端口。

## 解决方案

使用 Uvicorn 的 workers 参数：
- Supervisor 启动 1 个进程
- 该进程内部通过 `uvicorn --workers N` 启动 N 个 worker 进程
- 所有 worker 共享同一个端口（Uvicorn 自动处理端口复用）

## 需要修改的文件

### 1. `/workspace/supervisor/conf.d/glmocr-async.conf`

**当前状态：**
```ini
command=bash -c 'exec python -m glmocr.async_server --host 127.0.0.1 --port $(($GLMOCR_PORT + %(process_num)d))'
numprocs=4
```

**修改为：**
```ini
command=python -m glmocr.async_server --host 127.0.0.1 --port %(ENV_GLMOCR_PORT)s --workers 4
numprocs=1
```

**说明：**
- 移除 bash 脚本和端口偏移计算
- 添加 `--workers 4` 参数
- `numprocs=1` 表示只启动 1 个 Supervisor 进程

### 2. `/workspace/supervisor/conf.d/preprocess.conf`

**当前状态：**
```ini
command=bash -c 'exec python -m preprocess.server --host 127.0.0.1 --port $(($PREPROCESS_PORT + %(process_num)d))'
numprocs=2
```

**修改为：**
```ini
command=python -m preprocess.server --host 127.0.0.1 --port %(ENV_PREPROCESS_PORT)s --workers 2
numprocs=1
```

**说明：**
- 预处理服务使用 2 个 worker（GPU 模型加载较慢，worker 数不宜过多）

### 3. `/workspace/supervisor/conf.d/postprocess.conf`

**当前状态：**
```ini
command=bash -c 'exec python -m postprocess.server --host 127.0.0.1 --port $(($POSTPROCESS_PORT + %(process_num)d))'
numprocs=2
```

**修改为：**
```ini
command=python -m postprocess.server --host 127.0.0.1 --port %(ENV_POSTPROCESS_PORT)s --workers 2
numprocs=1
```

### 4. `/workspace/glmocr/async_server.py`

**当前状态：**
```python
parser.add_argument("--port", type=int, default=None, help="Port to bind to")
# 没有 --workers 参数

uvicorn.run(
    app,
    host=config.server.host,
    port=config.server.port,
    log_level=log_level.lower(),
)
```

**修改为：**
```python
parser.add_argument("--port", type=int, default=None, help="Port to bind to")
parser.add_argument("--workers", type=int, default=4, help="Number of Uvicorn worker processes")

uvicorn.run(
    app,
    host=config.server.host,
    port=config.server.port,
    log_level=log_level.lower(),
    workers=args.workers,
)
```

### 5. `/workspace/preprocess/server.py`

**当前状态：**
```python
def main():
    config = load_config()
    configure_logging(level=config.logging.level)
    
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower()
    )
```

**修改为：**
```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Preprocess Server")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    
    config = load_config()
    configure_logging(level=config.logging.level)
    
    # 覆盖配置
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower(),
        workers=args.workers,
    )
```

### 6. `/workspace/postprocess/server.py`

**当前状态：**
```python
def run_server():
    import uvicorn
    
    uvicorn.run(
        "postprocess.server:app",
        host=config.host,
        port=config.port,
        reload=config.debug
    )
```

**修改为：**
```python
import argparse

def run_server():
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Postprocess Server")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    
    # 覆盖配置
    host = args.host or config.host
    port = args.port or config.port
    
    uvicorn.run(
        "postprocess.server:app",
        host=host,
        port=port,
        reload=config.debug,
        workers=args.workers,
    )
```

## 验证步骤

1. **启动服务：**
   ```bash
   supervisord -c /workspace/supervisor/supervisord.conf
   supervisorctl -c /workspace/supervisor/supervisord.conf status
   ```

2. **检查端口监听：**
   ```bash
   netstat -tlnp | grep -E '(8000|8001|8002)'
   ```
   应该看到每个端口只有一个进程在监听（Uvicorn master process）

3. **测试并发请求：**
   使用测试脚本发送并发请求，验证多个 worker 是否能正常处理

4. **查看日志：**
   ```bash
   tail -f /var/log/supervisor/glmocr-async-00.log
   ```
   应该能看到多个 worker 进程的启动日志

## 优势

1. **简单：** 无需额外的 Nginx 层
2. **高效：** Uvicorn 自动处理端口复用和负载均衡
3. **灵活：** 通过调整 `--workers` 参数即可控制并发数
4. **生产就绪：** Uvicorn workers 是 FastAPI 官方推荐的生产部署方式

## 注意事项

1. **GPU 模型加载：** 每个 worker 进程都会独立加载 GPU 模型，内存占用会增加
2. **Worker 数量：** 
   - glmocr-async: 4 workers（CPU 密集型，可多配）
   - preprocess: 2 workers（GPU 模型较大，不宜过多）
   - postprocess: 2 workers（根据实际负载调整）
3. **监控：** 建议配合 Prometheus + Grafana 监控各 worker 的负载情况
