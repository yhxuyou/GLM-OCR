"""
任务管理模块
支持异步处理、队列管理和并发控制
"""
import asyncio
import uuid
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

from .config import get_settings
from .logger import logger


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """任务数据类"""
    task_id: str
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "started_at": datetime.fromtimestamp(self.started_at).isoformat() if self.started_at else None,
            "completed_at": datetime.fromtimestamp(self.completed_at).isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "duration": (self.completed_at - self.created_at) if self.completed_at else None
        }


class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        now = time.time()
        
        with self._lock:
            if client_id not in self.requests:
                self.requests[client_id] = []
            
            # 清理过期请求
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window_seconds
            ]
            
            # 检查是否超限
            if len(self.requests[client_id]) >= self.max_requests:
                return False
            
            self.requests[client_id].append(now)
            return True


class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=self.settings.MAX_QUEUE_SIZE)
        self.active_tasks: int = 0
        self._lock = asyncio.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._worker_tasks: List[asyncio.Task] = []
        self._shutdown_event: Optional[asyncio.Event] = None
        self.rate_limiter = RateLimiter(
            self.settings.RATE_LIMIT_REQUESTS,
            self.settings.RATE_LIMIT_WINDOW
        )
    
    async def initialize(self):
        """初始化任务管理器"""
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.MAX_CONCURRENT_TASKS,
            thread_name_prefix="glmocr_worker"
        )
        self._shutdown_event = asyncio.Event()
        
        # 启动worker协程
        for i in range(self.settings.MAX_CONCURRENT_TASKS):
            worker = asyncio.create_task(self._worker_loop(i))
            self._worker_tasks.append(worker)
        
        logger.info(f"TaskManager initialized with {self.settings.MAX_CONCURRENT_TASKS} workers")
    
    async def shutdown(self):
        """关闭任务管理器"""
        if self._shutdown_event:
            self._shutdown_event.set()
        
        # 等待所有worker结束
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # 关闭线程池
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
        
        logger.info("TaskManager shutdown complete")
    
    async def create_task(
        self,
        processor: Callable,
        metadata: Optional[Dict[str, Any]] = None,
        client_id: Optional[str] = None
    ) -> str:
        """创建新任务"""
        # 速率限制检查
        if self.settings.RATE_LIMIT_ENABLED and client_id:
            if not self.rate_limiter.is_allowed(client_id):
                raise RuntimeError("Rate limit exceeded")
        
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=time.time(),
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.tasks[task_id] = task
        
        # 加入队列
        try:
            await self.task_queue.put((task_id, processor))
            logger.info(f"Task {task_id} created and queued")
        except asyncio.QueueFull:
            async with self._lock:
                del self.tasks[task_id]
            raise RuntimeError("Task queue is full")
        
        return task_id
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        async with self._lock:
            return self.tasks.get(task_id)
    
    async def get_all_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """获取所有任务"""
        async with self._lock:
            tasks = list(self.tasks.values())
            if status:
                tasks = [t for t in tasks if t.status == status]
            return tasks
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task or task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
            
            task.status = TaskStatus.CANCELLED
            logger.info(f"Task {task_id} cancelled")
            return True
    
    async def _worker_loop(self, worker_id: int):
        """worker协程循环"""
        logger.info(f"Worker {worker_id} started")
        
        while not self._shutdown_event.is_set():
            try:
                # 尝试从队列获取任务，带超时
                try:
                    task_id, processor = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 处理任务
                await self._process_task(task_id, processor, worker_id)
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, task_id: str, processor: Callable, worker_id: int):
        """处理单个任务"""
        task = await self.get_task(task_id)
        if not task:
            return
        
        # 检查任务是否已取消
        if task.status == TaskStatus.CANCELLED:
            return
        
        async with self._lock:
            task.status = TaskStatus.PROCESSING
            task.started_at = time.time()
            self.active_tasks += 1
        
        try:
            # 在同步线程池中运行处理器
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, processor),
                timeout=self.settings.TASK_TIMEOUT
            )
            
            async with self._lock:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = time.time()
            
            logger.info(f"Task {task_id} completed by worker {worker_id}")
            
        except asyncio.TimeoutError:
            async with self._lock:
                task.status = TaskStatus.TIMEOUT
                task.error = "Task timeout"
                task.completed_at = time.time()
            logger.error(f"Task {task_id} timed out")
            
        except Exception as e:
            async with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = time.time()
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            
        finally:
            async with self._lock:
                self.active_tasks -= 1
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        async with self._lock:
            return {
                "queue_size": self.task_queue.qsize(),
                "max_queue_size": self.settings.MAX_QUEUE_SIZE,
                "active_tasks": self.active_tasks,
                "max_concurrent_tasks": self.settings.MAX_CONCURRENT_TASKS,
                "total_tasks": len(self.tasks)
            }


# 全局任务管理器实例
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """获取任务管理器单例"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
