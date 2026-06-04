# 图像预处理 4-进程池 + 异步 OCR + 文档聚合 实施计划

## 1. 摘要 (Summary)

新增一个独立模块 `glmocr/preprocess_pool/`，对外提供「**4 进程图像预处理 → 异步 OCR → 按文档聚合结果**」的端到端能力：

- 复用 SDK 已有的 `OCRClient`（HTTP 调用 + 重试 + 连接池）和 `PageLoader.build_request_from_image`（构造 OCR 请求），不重新实现 vLLM 协议。
- **不**走 `Pipeline.process()`（它内置 layout detection，会与用户的预处理重复）。
- 预处理在 4 个独立进程中跑（CPU 密集 + 用户自定义模型）；OCR 在主进程内的 `ThreadPoolExecutor` 中并发跑（I/O 密集）。
- 同一 `document_id` 的所有 region OCR 完成后，通过用户提供的回调一次性交付。

---

## 2. 现状分析 (Current State Analysis)

### 2.1 已读过的关键文件

| 文件 | 关键信息 |
|------|---------|
| [pipeline/pipeline.py](file:///workspace/glmocr/pipeline/pipeline.py) | 三线程串行 pipeline（load → layout → recognition），用 `PipelineState` + `UnitTracker` 协调；不满足「异步拿结果 + 按文档聚合」的需求。 |
| [pipeline/_workers.py](file:///workspace/glmocr/pipeline/_workers.py) | 区域级别并行 OCR 的实现范式可借鉴：line 318 `recognition_worker` 用 `ThreadPoolExecutor` + futures dict + `as_completed`。 |
| [pipeline/_state.py](file:///workspace/glmocr/pipeline/_state.py) | 跨线程共享状态的范式：queue + 锁 + 后台 `shutdown_event`。 |
| [pipeline/_unit_tracker.py](file:///workspace/glmocr/pipeline/_unit_tracker.py) | 「unit 完成时通知主线程」的最小实现。 |
| [ocr_client.py](file:///workspace/glmocr/ocr_client.py) | 同步 HTTP `requests.Session`（含重试 / 连接池 / OpenAI & Ollama 协议）；`process()` 返回 `(response_dict, status_code)`。直接复用。 |
| [dataloader/page_loader.py](file:///workspace/glmocr/dataloader/page_loader.py) | `build_request_from_image(image, task_type)` 在 line 340 提供了从 PIL Image 构造 OCR 请求 payload 的标准方法（含 prompt mapping、resize、base64 编码）。直接复用。 |
| [parser_result/pipeline_result.py](file:///workspace/glmocr/parser_result/pipeline_result.py) | 最终结果对象的结构：`json_result` / `markdown_result` / `original_images` / `image_files` / `layout_vis_images`。 |
| [config.py](file:///workspace/glmocr/config.py) | `PipelineConfig.ocr_api` 已经包含了本地 vLLM 所需的全部参数；只需新增一个 `preprocess_pool` 子配置块。 |

### 2.2 现有架构的不足

1. `GlmOcr.parse()` 是「同步阻塞」语义：调用方必须 `for r in parser.parse(...)` 等所有结果。
2. `Pipeline.process()` 把 layout detection 写死在 stage 2；用户已经在做自己的 layout，没法跳过。
3. 结果按 `unit_idx` 顺序 yield（preserve_order=True）或乱序 yield，没有「等一个文档全部 region OCR 完再触发回调」的语义。

### 2.3 需要新增/修改的文件

全部新增，**不修改** SDK 现有代码（保证向后兼容）。

---

## 3. 拟新增模块 (Proposed Changes)

新目录：`glmocr/preprocess_pool/`

```
glmocr/preprocess_pool/
├── __init__.py              # 公开 API
├── protocols.py             # DocumentPreprocessor / Region 数据契约
├── tasks.py                 # PreprocessTask / RegionTask 数据类
├── preprocess_worker.py     # 跑在子进程里的 worker 入口
├── async_ocr.py             # OCRClient 的 ThreadPoolExecutor 包装
├── aggregator.py            # 按 document_id 聚合 region + 触发回调
└── pool.py                  # PreprocessPool 主体（4 进程池 + 编排）
```

同时：
- `glmocr/config.py`：新增 `PreprocessPoolConfig` pydantic 模型并接入 `PipelineConfig.preprocess_pool`。
- `glmocr/config.yaml`：添加对应的可调参数块（默认 4 进程，OCR 并发 32，队列大小可配）。
- `glmocr/__init__.py`：在 `_LAZY_ATTRS` 中注册 `PreprocessPool`（延迟加载，避免冷启动开销）。

---

### 3.1 `protocols.py` — 数据契约

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
from PIL import Image


@dataclass
class Region:
    """预处理产出的单个识别区域。"""
    image: Image.Image            # 已裁剪好的 PIL 图
    bbox: tuple                   # 原始图坐标 (x1, y1, x2, y2)，像素
    task_type: str                # "text" / "table" / "formula" / "skip" / "abandon"
    label: str = ""               # 可选：来自 layout 的 label
    polygon: Optional[list] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentPreprocessor(Protocol):
    """用户实现的预处理接口（4 步：文档检测/方向/扭曲矫正/布局检测）。

    实例化代价高（要加载模型），因此会作为工厂传入，每个子进程各自调用一次。
    """
    def process(self, image: Image.Image, **kwargs: Any) -> List[Region]:
        ...
```

`process()` 的内部实现由用户写（导入自己的 doc 检测 / 方向 / 扭曲 / 布局模型）。`abandon` 类型的 region 不会进入 OCR 队列；`skip` 类型不进 OCR 但会被记录到最终结果（图片保留）。

---

### 3.2 `tasks.py` — 跨进程消息

```python
@dataclass
class PreprocessTask:
    task_id: str
    document_id: str
    image_bytes: bytes            # PNG 编码，跨进程传递
    on_done: Optional[Callable]   # 注意：回调对象本身在子进程里无法 pickle
                                   # 所以 on_done 只在主进程使用，子进程置 None
    preprocess_kwargs: Dict[str, Any]


@dataclass
class RegionTask:
    region_id: str                # f"{task_id}#{idx}"
    document_id: str
    task_id: str
    image_bytes: bytes            # PNG
    bbox: tuple
    task_type: str
    label: str
    polygon: Optional[list]
    on_done: Optional[Callable]   # 子进程里必为 None
    metadata: Dict[str, Any]


@dataclass
class RegionError:
    region_id: str
    document_id: str
    error: str


# 子进程退出哨兵
WORKER_STOP_SENTINEL = None
```

**为什么用 bytes 而不是 PIL Image 跨进程**：`multiprocessing.Queue` 序列化 PIL Image 很慢，PNG 字节流更稳。代价是多一次 encode/decode 往返。

---

### 3.3 `preprocess_worker.py` — 子进程入口

```python
import io
from PIL import Image
from multiprocessing.queues import Queue as MpQueue
from glmocr.preprocess_pool.tasks import (
    PreprocessTask, RegionTask, RegionError, WORKER_STOP_SENTINEL,
)
from glmocr.preprocess_pool.protocols import DocumentPreprocessor
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


def preprocess_worker(
    input_q: MpQueue,
    output_q: MpQueue,
    preprocessor_factory: Callable[[], DocumentPreprocessor],
) -> None:
    """跑在子进程里的主循环。每个进程只实例化一次 Preprocessor。"""
    preprocessor = preprocessor_factory()
    logger.info("Preprocess worker ready (pid=%d)", os.getpid())

    while True:
        item: PreprocessTask = input_q.get()
        if item is WORKER_STOP_SENTINEL:
            break
        try:
            image = Image.open(io.BytesIO(item.image_bytes))
            regions = preprocessor.process(image, **item.preprocess_kwargs)
        except Exception as e:
            logger.exception("Preprocess failed for task %s", item.task_id)
            output_q.put(RegionError(
                region_id=item.task_id,
                document_id=item.document_id,
                error=f"preprocess: {e}",
            ))
            continue

        for idx, region in enumerate(regions):
            buf = io.BytesIO()
            region.image.save(buf, format="PNG")
            output_q.put(RegionTask(
                region_id=f"{item.task_id}#{idx}",
                document_id=item.document_id,
                task_id=item.task_id,
                image_bytes=buf.getvalue(),
                bbox=region.bbox,
                task_type=region.task_type,
                label=region.label,
                polygon=region.polygon,
                on_done=None,  # 子进程里回调传不回去，靠 main 进程聚合
                metadata=region.metadata,
            ))
```

---

### 3.4 `async_ocr.py` — OCR 异步派发

```python
import io
import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Callable, Dict, Optional
from PIL import Image

from glmocr.ocr_client import OCRClient
from glmocr.dataloader import PageLoader
from glmocr.preprocess_pool.tasks import RegionTask, RegionError
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class AsyncOCRDispatcher:
    """包装同步 OCRClient，用 ThreadPoolExecutor 实现「提交即返回」。

    设计要点：
    - 复用 OCRClient 的 requests.Session（含连接池 + 重试 + Ollama 协议转换）。
    - 复用 PageLoader.build_request_from_image（统一 prompt mapping / resize / base64）。
    - futures dict 追踪 in-flight 请求，结果按完成顺序回流到回调。
    """
    def __init__(
        self,
        ocr_client: OCRClient,
        page_loader: PageLoader,
        max_workers: int = 32,
        on_region_done: Optional[Callable[[RegionTask, dict], None]] = None,
    ):
        self.ocr_client = ocr_client
        self.page_loader = page_loader
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ocr-async",
        )
        self.futures: Dict[Future, RegionTask] = {}
        self.on_region_done = on_region_done
        self._closed = False

    def submit(self, region_task: RegionTask) -> None:
        if self._closed:
            raise RuntimeError("AsyncOCRDispatcher is closed")
        # abandon 类型直接进 aggregator（无 content）
        if region_task.task_type == "abandon":
            if self.on_region_done:
                self.on_region_done(region_task, {"content": None, "status": 204})
            return
        # skip 类型：图片保留，但不需要 OCR
        if region_task.task_type == "skip":
            if self.on_region_done:
                self.on_region_done(region_task, {"content": None, "status": 200, "skip": True})
            return

        image = Image.open(io.BytesIO(region_task.image_bytes))
        req = self.page_loader.build_request_from_image(image, region_task.task_type)
        future = self.executor.submit(self.ocr_client.process, req)
        self.futures[future] = region_task

    def drain(self, timeout: float = 0.05) -> int:
        """回收已完成的 future，返回本次回收数量。"""
        if not self.futures:
            return 0
        done = [f for f in self.futures if f.done()]
        for f in done:
            region_task = self.futures.pop(f)
            try:
                response, status_code = f.result()
                payload = {"response": response, "status": status_code}
            except Exception as e:
                logger.warning("OCR failed for region %s: %s", region_task.region_id, e)
                payload = {"response": None, "status": 500, "error": str(e)}
            if self.on_region_done:
                self.on_region_done(region_task, payload)
        return len(done)

    def has_inflight(self) -> bool:
        return bool(self.futures)

    def close(self) -> None:
        self._closed = True
        self.executor.shutdown(wait=True)
```

**为什么是 `on_region_done` 回调而不是阻塞 get**：让 `drain()` 可以在主循环里被反复短轮询（不阻塞线程），与 `_workers.py:431 _wait_for_any` 范式一致。

---

### 3.5 `aggregator.py` — 按文档聚合

```python
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from glmocr.preprocess_pool.tasks import RegionTask


@dataclass
class _PendingDoc:
    document_id: str
    expected: int                # 总 region 数
    received: int = 0
    regions: List[Dict[str, Any]] = field(default_factory=list)
    on_done: Optional[Callable] = None
    source_image_path: Optional[str] = None


class DocumentAggregator:
    """记录每个 document 期待收到的 region 数；收齐后一次性回调。

    计数策略：每次 submit 时先注册期望（预期 region 数由用户传，或在 preprocess
    完成回填时再 finalize）。简化做法：用户传 `expected_regions=None` 表示「有多少
    收多少，回调由用户在外部触发」；传整数则由 aggregator 自动 finalize。
    """
    def __init__(self):
        self._pending: Dict[str, _PendingDoc] = {}

    def register(
        self,
        document_id: str,
        expected_regions: Optional[int],
        on_done: Callable[[Dict[str, Any]], None],
        source_image_path: Optional[str] = None,
    ) -> None:
        self._pending[document_id] = _PendingDoc(
            document_id=document_id,
            expected=expected_regions or 0,
            on_done=on_done,
            source_image_path=source_image_path,
        )

    def on_region(self, region_task: RegionTask, payload: Dict[str, Any]) -> Optional[Dict]:
        """收到一个 region 的 OCR 结果。

        返回值：若整个文档收齐，返回最终的聚合 dict；否则返回 None。
        """
        doc = self._pending.get(region_task.document_id)
        if doc is None:
            # 用户没 register；忽略（避免崩溃）
            return None

        # 情况 A：用户事先不知道 region 数 → 第一次收到 region 时把 expected 设为 ∞
        if doc.expected == 0:
            # 等所有 region 都回来时由用户主动调用 finalize（不实现也行）
            pass

        doc.regions.append({
            "region_id": region_task.region_id,
            "task_id": region_task.task_id,
            "bbox": region_task.bbox,
            "task_type": region_task.task_type,
            "label": region_task.label,
            "polygon": region_task.polygon,
            "metadata": region_task.metadata,
            "response": payload.get("response"),
            "status": payload.get("status"),
            "skip": payload.get("skip", False),
        })
        doc.received += 1

        if doc.expected and doc.received >= doc.expected:
            return self._finalize(doc)
        return None

    def _finalize(self, doc: _PendingDoc) -> Dict[str, Any]:
        result = {
            "document_id": doc.document_id,
            "source_image_path": doc.source_image_path,
            "regions": doc.regions,
            "num_regions": len(doc.regions),
        }
        if doc.on_done:
            try:
                doc.on_done(result)
            except Exception as e:
                logger.exception("on_done callback raised: %s", e)
        self._pending.pop(doc.document_id, None)
        return result

    def has_pending(self) -> bool:
        return bool(self._pending)
```

> 备注：用户场景下「一个文档有多少 region」通常取决于预处理输出（多步布局的 PDF），所以推荐 `expected_regions=None` + 用户拿到 region 列表后自己判完成。也可以扩展一个 `flush_document(doc_id)` 接口给用户主动 finalize。

---

### 3.6 `pool.py` — 4 进程池主体

```python
import io
import os
import threading
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue as MpQueue
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Protocol

from glmocr.config import GlmOcrConfig, PipelineConfig
from glmocr.dataloader import PageLoader
from glmocr.ocr_client import OCRClient
from glmocr.preprocess_pool.async_ocr import AsyncOCRDispatcher
from glmocr.preprocess_pool.aggregator import DocumentAggregator
from glmocr.preprocess_pool.preprocess_worker import preprocess_worker
from glmocr.preprocess_pool.protocols import DocumentPreprocessor
from glmocr.preprocess_pool.tasks import (
    PreprocessTask, RegionTask, RegionError, WORKER_STOP_SENTINEL,
)
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class _SubContext:
    """跟踪一次 submit 的元数据，跨子进程不会传出去（子进程置 None）。"""
    document_id: str
    expected_regions: Optional[int]
    on_done: Optional[Callable[[Dict[str, Any]], None]]
    source_image_path: Optional[str]


class PreprocessPool:
    """用户入口。

    用法::

        from glmocr import GlmOcrConfig, PreprocessPool

        class MyPreprocessor:
            def process(self, image, **_):
                # 1) 文档检测
                # 2) 方向检测
                # 3) 扭曲矫正
                # 4) 布局检测
                return regions  # List[Region]

        cfg = GlmOcrConfig.from_env()
        pool = PreprocessPool(
            preprocessor=MyPreprocessor,
            config=cfg,
        )
        pool.start()

        def on_doc_done(result):
            print("doc finished:", result["document_id"], "regions:", result["num_regions"])

        pool.submit(
            image_path="doc.png",
            document_id="doc-001",
            on_done=on_doc_done,
        )
        pool.join()
        pool.stop()
    """

    def __init__(
        self,
        preprocessor: Callable[[], DocumentPreprocessor],
        config: GlmOcrConfig,
        num_workers: int = 4,
        ocr_max_workers: int = 32,
        page_loader: Optional[PageLoader] = None,
        ocr_client: Optional[OCRClient] = None,
    ):
        self._preprocessor_factory = preprocessor
        self._config = config
        self._num_workers = num_workers
        self._ocr_max_workers = ocr_max_workers

        self._ocr_client = ocr_client or OCRClient(config.pipeline.ocr_api)
        self._page_loader = page_loader or PageLoader(config.pipeline.page_loader)

        self._input_q: Optional[MpQueue] = None
        self._output_q: Optional[MpQueue] = None
        self._processes: List[Process] = []
        self._dispatcher: Optional[AsyncOCRDispatcher] = None
        self._aggregator: Optional[DocumentAggregator] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 跟踪已 submit 的 task/document 元数据
        self._sub_ctx: Dict[str, _SubContext] = {}     # task_id → ctx
        self._doc_subtasks: Dict[str, List[str]] = {}  # document_id → [task_id]
        self._lock = threading.Lock()

    # ---- 生命周期 ----
    def start(self) -> None:
        if self._processes:
            raise RuntimeError("Pool already started")
        self._ocr_client.start()
        self._input_q = MpQueue(maxsize=200)
        self._output_q = MpQueue(maxsize=2000)
        for i in range(self._num_workers):
            p = Process(
                target=preprocess_worker,
                args=(self._input_q, self._output_q, self._preprocessor_factory),
                name=f"preprocess-{i}",
                daemon=True,
            )
            p.start()
            self._processes.append(p)
        logger.info("PreprocessPool started with %d workers", self._num_workers)

        self._aggregator = DocumentAggregator()
        self._dispatcher = AsyncOCRDispatcher(
            ocr_client=self._ocr_client,
            page_loader=self._page_loader,
            max_workers=self._ocr_max_workers,
            on_region_done=self._on_region_done,
        )

        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._result_reader_loop, name="pool-reader", daemon=True
        )
        self._reader_thread.start()

    def stop(self) -> None:
        if not self._processes:
            return
        for _ in self._processes:
            self._input_q.put(WORKER_STOP_SENTINEL)
        for p in self._processes:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning("Process %s did not exit in time, terminating", p.name)
                p.terminate()
        self._processes.clear()

        self._stop_event.set()
        if self._reader_thread:
            self._reader_thread.join(timeout=5)
        if self._dispatcher:
            self._dispatcher.close()
        self._ocr_client.stop()
        logger.info("PreprocessPool stopped")

    # ---- 提交 ----
    def submit(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        pil_image: Optional[Image.Image] = None,
        document_id: str = "",
        task_id: Optional[str] = None,
        on_done: Optional[Callable[[Dict[str, Any]], None]] = None,
        expected_regions: Optional[int] = None,
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """提交一张图（路径/字节/PIL 三选一）进行预处理。返回 task_id。"""
        if image_path:
            with open(image_path, "rb") as f:
                data = f.read()
        elif image_bytes is not None:
            data = image_bytes
        elif pil_image is not None:
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            data = buf.getvalue()
        else:
            raise ValueError("必须提供 image_path / image_bytes / pil_image 之一")

        task_id = task_id or f"t-{int(time.time()*1e6)}-{os.getpid()}"

        with self._lock:
            self._sub_ctx[task_id] = _SubContext(
                document_id=document_id,
                expected_regions=expected_regions,
                on_done=on_done,
                source_image_path=image_path,
            )
            self._doc_subtasks.setdefault(document_id, []).append(task_id)
            # 注册聚合（首次见到这个 document 时注册回调）
            if on_done is not None and document_id not in self._aggregator._pending:
                self._aggregator.register(
                    document_id=document_id,
                    expected_regions=expected_regions,
                    on_done=on_done,
                    source_image_path=image_path,
                )

        self._input_q.put(PreprocessTask(
            task_id=task_id,
            document_id=document_id,
            image_bytes=data,
            on_done=None,
            preprocess_kwargs=preprocess_kwargs or {},
        ))
        return task_id

    # ---- 后台循环 ----
    def _result_reader_loop(self) -> None:
        """不停从子进程 output_q 取 region，分发给 AsyncOCRDispatcher。"""
        while not self._stop_event.is_set():
            try:
                item = self._output_q.get(timeout=0.1)
            except Exception:
                # queue.Empty 的兼容写法
                self._dispatcher.drain()
                continue

            if isinstance(item, RegionError):
                # 整张图预处理失败 → 模拟一个空 region 让 aggregator 推进
                logger.error("RegionError: %s", item.error)
                self._aggregator.on_region(
                    RegionTask(
                        region_id=item.region_id,
                        document_id=item.document_id,
                        task_id=item.region_id,
                        image_bytes=b"",
                        bbox=(0, 0, 0, 0),
                        task_type="abandon",
                        label="",
                        polygon=None,
                        on_done=None,
                        metadata={"error": item.error},
                    ),
                    {"content": None, "status": 500, "error": item.error},
                )
                continue

            if isinstance(item, RegionTask):
                self._dispatcher.submit(item)

            # 顺手回收 OCR 完成项
            self._dispatcher.drain()

    def _on_region_done(self, region_task: RegionTask, payload: Dict[str, Any]) -> None:
        # Aggregator 内部会判断是否收齐，并触发 on_done 回调
        self._aggregator.on_region(region_task, payload)

    # ---- 阻塞等待 ----
    def join(self, timeout: Optional[float] = None) -> None:
        """阻塞直到所有已 submit 的 region 都被 OCR 完。"""
        deadline = time.time() + timeout if timeout else None
        while True:
            inflight = (
                (self._input_q.qsize() if self._input_q else 0)
                + (self._output_q.qsize() if self._output_q else 0)
                + (self._dispatcher.has_inflight() if self._dispatcher else 0)
            )
            if inflight == 0 and not self._aggregator.has_pending():
                return
            if deadline and time.time() > deadline:
                logger.warning("join() timed out, %d items still in flight", inflight)
                return
            time.sleep(0.1)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
```

---

### 3.7 `config.py` / `config.yaml` 扩展

```python
class PreprocessPoolConfig(_BaseConfig):
    num_workers: int = 4                # 预处理子进程数
    ocr_max_workers: int = 32           # OCR 异步线程数
    input_queue_maxsize: int = 200
    output_queue_maxsize: int = 2000
    join_poll_interval: float = 0.1
```

接入 `PipelineConfig`：

```python
class PipelineConfig(_BaseConfig):
    ...
    preprocess_pool: PreprocessPoolConfig = Field(default_factory=PreprocessPoolConfig)
```

`config.yaml` 增加：

```yaml
pipeline:
  preprocess_pool:
    num_workers: 4
    ocr_max_workers: 32
    input_queue_maxsize: 200
    output_queue_maxsize: 2000
```

---

### 3.8 `__init__.py` 暴露

```python
from .pool import PreprocessPool
from .protocols import DocumentPreprocessor, Region
```

并在 `glmocr/__init__.py` 的 `_LAZY_ATTRS` 中：

```python
"PreprocessPool": ("preprocess_pool", "PreprocessPool"),
"DocumentPreprocessor": ("preprocess_pool.protocols", "DocumentPreprocessor"),
"Region": ("preprocess_pool.protocols", "Region"),
```

---

## 4. 假设与决策 (Assumptions & Decisions)

| 假设 | 备注 |
|------|------|
| **A1**：用户的预处理模型在 4 个进程里各自加载一份。 | 多 GPU 时用户自行在 `preprocessor_factory` 里给每个进程绑不同 `cuda_visible_devices`。 |
| **A2**：「一个文档有多少 region」由用户在 `submit(..., expected_regions=N)` 显式声明；不声明时 aggregator 不会自动 finalize，由用户基于 `on_region_done` 自管完成判断。 | 简化「region 数量未知」时的语义，避免漏回调。 |
| **A3**：跨进程用 PNG 字节流，不用 pickle 整个 PIL Image。 | 稳定性更好，代价是一两次 encode/decode。 |
| **A4**：`OCRClient` 实例在主进程创建并 `start()`；每个 OCR 线程共享同一 `requests.Session`（连接池 + 重试自然生效）。 | 复用 [ocr_client.py:105](file:///workspace/glmocr/ocr_client.py#L105) `_make_session`。 |
| **A5**：`skip` 类型 region 把 cropped image 暂存在 `region_task.metadata["image_bytes"]` 里，aggregator 在回调里把它转成 PIL Image 交给 `PipelineResult.image_files`。 | 与 [pipeline_result.py:36](file:///workspace/glmocr/parser_result/pipeline_result.py#L36) `image_files` 字段对齐。 |
| **A6**：子进程崩溃时由主进程 `reader_loop` 探测 `Process.is_alive()`，并尝试重启（最多 3 次）；最终失败则把该 task 标 RegionError 推回。 | 容错。 |
| **D1**：不修改 `Pipeline.process()`、`GlmOcr.parse()` 等现有 SDK 入口。新模块完全独立。 | 避免破坏向后兼容。 |
| **D2**：OCR 异步用 `concurrent.futures.ThreadPoolExecutor`，不用 `asyncio + aiohttp`。 | 用户已确认；改动最小，重用现有重试/连接池。 |
| **D3**：结果交付用回调（`on_done`），不在主进程暴露 Queue。 | 用户已确认。 |
| **D4**：布局检测由用户自己实现。模块只定义 `DocumentPreprocessor` 协议。 | 用户已确认。 |

---

## 5. 验证步骤 (Verification)

实施后按以下顺序验证：

1. **单元冒烟**：`tests/test_preprocess_pool.py`
   - mock `DocumentPreprocessor` 返回 1 / 5 / 20 个 region，验证：
     - `submit → join` 闭环正常；
     - OCR 失败时 region 的 `status=500` 被回填到 `result["regions"]`；
     - `abandon` / `skip` 类型的 region 路径；
     - 4 个子进程都被实际用到（通过 `os.getpid()` 验证 diversity）。

2. **集成**：`tests/test_preprocess_pool_integration.py`
   - 用一个真实轻量 `DocumentPreprocessor`（如返回固定 bbox 的占位实现）；
   - 启动一个 mock HTTP server（`http.server` + `threading`），模拟 vLLM `/v1/chat/completions`；
   - 验证：100 张图并行 submit，最终 `num_regions == 100 * regions_per_image`，回调被精确触发一次。

3. **健壮性**：
   - OCR server 故意 503 → 验证 SDK 现有重试生效，region 最终 `status=200`；
   - 杀掉 1 个子进程 → 验证 reader_loop 探测并重启；
   - 提交 1000 张图时检查 `multiprocessing.Queue` 没有 OOM（PNG 字节流大小可控）。

4. **运行命令**（需要时）：
   ```bash
   cd /workspace
   pytest glmocr/preprocess_pool/tests/ -v
   ```

5. **不破坏现有**：跑 `glmocr/tests/test_unit.py` + `test_integration.py`，确保 `GlmOcr.parse()`、`Pipeline.process()` 行为不变。

---

## 6. 实施顺序（建议）

1. `protocols.py` + `tasks.py`（最纯的 dataclass，无依赖）
2. `preprocess_worker.py`（先打通子进程通路）
3. `async_ocr.py`（先把 `OCRClient` 异步化）
4. `aggregator.py`（聚合逻辑）
5. `pool.py`（串起来）
6. `config.py` + `config.yaml` 扩展
7. `__init__.py` 暴露
8. 单测 + 集成测试
