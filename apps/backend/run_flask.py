#!/usr/bin/env python
"""
Flask服务启动脚本
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app_flask import run_flask_app

if __name__ == "__main__":
    run_flask_app()
