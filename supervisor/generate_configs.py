#!/usr/bin/env python3
"""
Supervisor 配置文件生成器

根据指定的进程数量生成 Supervisor 配置文件，自动分配端口。
"""

import argparse
import os
from pathlib import Path

TEMPLATE_PREPROCESS = """; 预处理服务 - 自动生成配置
; 进程数: {num_procs}
; 端口范围: {port_start} - {port_end}

[program:preprocess-service]
command=python -m preprocess.server --host 127.0.0.1 --port %(ENV_PREPROCESS_PORT_{process_num}d)s
directory={workspace}
process_name=%(program_name)s_{process_num:02d}
numprocs=1
autostart=true
autorestart=true
startsecs=10
startretries=3
stopwaitsecs=30
redirect_stderr=true
stdout_logfile={log_dir}/preprocess-{process_num:02d}.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
environment=
    PREPROCESS_PORT="%(ENV_PREPROCESS_PORT_{process_num}d)s",
    PREPROCESS_HOST="127.0.0.1",
    PREPROCESS_DEVICE="cuda:{device_id}",
    PYTHONUNBUFFERED="1"
user={user}
priority=100
"""

TEMPLATE_GLMOCR = """; GLM OCR 异步服务 - 自动生成配置
; 进程数: {num_procs}
; 端口范围: {port_start} - {port_end}

[program:glmocr-async-service]
command=python -m glmocr.async_server --host 127.0.0.1 --port %(ENV_GLMOCR_PORT_{process_num}d)s
directory={workspace}
process_name=%(program_name)s_{process_num:02d}
numprocs=1
autostart=true
autorestart=true
startsecs=15
startretries=3
stopwaitsecs=60
redirect_stderr=true
stdout_logfile={log_dir}/glmocr-async-{process_num:02d}.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=20
environment=
    GLMOCR_PORT="%(ENV_GLMOCR_PORT_{process_num}d)s",
    GLMOCR_HOST="127.0.0.1",
    PYTHONUNBUFFERED="1",
    REDIS_URL="{redis_url}"
user={user}
priority=200
"""

TEMPLATE_POSTPROCESS = """; 后处理服务 - 自动生成配置
; 进程数: {num_procs}
; 端口范围: {port_start} - {port_end}

[program:postprocess-service]
command=python -m postprocess.server --host 127.0.0.1 --port %(ENV_POSTPROCESS_PORT_{process_num}d)s
directory={workspace}
process_name=%(program_name)s_{process_num:02d}
numprocs=1
autostart=true
autorestart=true
startsecs=10
startretries=3
stopwaitsecs=30
redirect_stderr=true
stdout_logfile={log_dir}/postprocess-{process_num:02d}.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
environment=
    POSTPROCESS_PORT="%(ENV_POSTPROCESS_PORT_{process_num}d)s",
    POSTPROCESS_HOST="127.0.0.1",
    PYTHONUNBUFFERED="1"
user={user}
priority=300
"""


def generate_service_config(
    service_name: str,
    num_procs: int,
    port_start: int,
    template: str,
    output_dir: Path,
    **kwargs
) -> list:
    """生成单个服务的配置"""
    ports = []
    config_content = []

    for i in range(num_procs):
        port = port_start + i
        ports.append(port)

        config = template.format(
            num_procs=num_procs,
            port_start=port_start,
            port_end=port_start + num_procs - 1,
            process_num=i,
            **kwargs
        )
        config_content.append(config)

    # 写入配置文件
    output_file = output_dir / f"{service_name}.conf"
    with open(output_file, 'w') as f:
        f.write('\n'.join(config_content))

    return ports


def main():
    parser = argparse.ArgumentParser(description='生成 Supervisor 配置文件')
    parser.add_argument('--preprocess-procs', type=int, default=2, help='预处理服务进程数')
    parser.add_argument('--glmocr-procs', type=int, default=4, help='GLM OCR 服务进程数')
    parser.add_argument('--postprocess-procs', type=int, default=2, help='后处理服务进程数')
    parser.add_argument('--preprocess-port-start', type=int, default=5003, help='预处理服务起始端口')
    parser.add_argument('--glmocr-port-start', type=int, default=8000, help='GLM OCR 服务起始端口')
    parser.add_argument('--postprocess-port-start', type=int, default=5005, help='后处理服务起始端口')
    parser.add_argument('--workspace', type=str, default='/workspace', help='工作目录')
    parser.add_argument('--log-dir', type=str, default='/var/log/supervisor', help='日志目录')
    parser.add_argument('--user', type=str, default='root', help='运行用户')
    parser.add_argument('--redis-url', type=str, default='redis://localhost:6379', help='Redis URL')
    parser.add_argument('--output-dir', type=str, default='/workspace/supervisor/conf.d', help='输出目录')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"生成 Supervisor 配置文件...")
    print(f"  预处理服务: {args.preprocess_procs} 个进程, 端口 {args.preprocess_port_start}-{args.preprocess_port_start + args.preprocess_procs - 1}")
    print(f"  GLM OCR 服务: {args.glmocr_procs} 个进程, 端口 {args.glmocr_port_start}-{args.glmocr_port_start + args.glmocr_procs - 1}")
    print(f"  后处理服务: {args.postprocess_procs} 个进程, 端口 {args.postprocess_port_start}-{args.postprocess_port_start + args.postprocess_procs - 1}")

    # 生成预处理服务配置
    preprocess_ports = generate_service_config(
        'preprocess',
        args.preprocess_procs,
        args.preprocess_port_start,
        TEMPLATE_PREPROCESS,
        output_dir,
        workspace=args.workspace,
        log_dir=args.log_dir,
        user=args.user,
        device_id=0
    )

    # 生成 GLM OCR 服务配置
    glmocr_ports = generate_service_config(
        'glmocr-async',
        args.glmocr_procs,
        args.glmocr_port_start,
        TEMPLATE_GLMOCR,
        output_dir,
        workspace=args.workspace,
        log_dir=args.log_dir,
        user=args.user,
        redis_url=args.redis_url
    )

    # 生成后处理服务配置
    postprocess_ports = generate_service_config(
        'postprocess',
        args.postprocess_procs,
        args.postprocess_port_start,
        TEMPLATE_POSTPROCESS,
        output_dir,
        workspace=args.workspace,
        log_dir=args.log_dir,
        user=args.user
    )

    print(f"\n配置文件已生成到: {output_dir}")
    print(f"\n端口分配:")
    print(f"  预处理服务: {preprocess_ports}")
    print(f"  GLM OCR 服务: {glmocr_ports}")
    print(f"  后处理服务: {postprocess_ports}")

    # 生成环境变量文件
    env_file = output_dir / 'service_ports.env'
    with open(env_file, 'w') as f:
        f.write("# 服务端口配置 - 自动生成\n")
        f.write(f"# 生成时间: {__import__('datetime').datetime.now()}\n\n")

        for i, port in enumerate(preprocess_ports):
            f.write(f"PREPROCESS_PORT_{i}={port}\n")
        for i, port in enumerate(glmocr_ports):
            f.write(f"GLMOCR_PORT_{i}={port}\n")
        for i, port in enumerate(postprocess_ports):
            f.write(f"POSTPROCESS_PORT_{i}={port}\n")

    print(f"\n环境变量文件: {env_file}")


if __name__ == '__main__':
    main()
