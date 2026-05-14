# GLM-OCR FastAPI服务 Dockerfile
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip

# 安装GLM-OCR（使用项目根目录的pyproject.toml）
RUN pip install --no-cache-dir -e ".[all]"

# 安装FastAPI部署依赖
RUN pip install --no-cache-dir -r deploy/fastapi/requirements.txt

# 创建必要的目录
RUN mkdir -p /tmp/glmocr_uploads

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "deploy.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
