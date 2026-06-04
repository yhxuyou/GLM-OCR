# 把 4 步预处理拆成 4 个可组合的检测器

## 1. 摘要 (Summary)

把当前 `MyDocPreprocessor` 的 4 个步骤拆成 4 个独立、可插拔的检测器类：

| 步骤 | 角色 | 实现方 |
|------|------|--------|
| 文档检测 (Document detection) | `BaseDocumentDetector` | 用户自己实现 |
| 方向检测 (Orientation detection) | `BaseOrientationDetector` | 用户自己实现 |
| 扭曲矫正 (Distortion correction) | `BaseDistortionCorrector` | 用户自己实现 |
| 布局检测 (Layout detection) | `LayoutDetectorAdapter` | **复用现有** `BaseLayoutDetector` / `PPDocLayoutDetector`（零代码） |

新模块位于 `glmocr/preprocess_pool/detectors/`，每个 ABC 提供统一的 **init / start / stop / detect** 生命周期，以及「**古典 CV 预处理 + NotImplementedError 推理**」骨架，用户的代码改动只剩 `_load_model()` 和 `_run_inference()` 两处。

新加 `DocumentPreprocessorPipeline` 把 4 个检测器串成一个标准 pipeline，**它本身就是一个 `DocumentPreprocessor`**，可以直接喂给 `PreprocessPool`。

> 注意：原 `protocols.py` 里的 `DocumentPreprocessor` Protocol **保留** —— 用户如果想完全自己编排，仍然可以实现该 Protocol。

---

## 2. 现状分析 (Current State Analysis)

### 2.1 关键文件
- [layout_detector.py](file:///workspace/glmocr/layout/layout_detector.py) — `PPDocLayoutDetector`，`process()` 返回 `List[List[Dict]]`，每条 dict 是 `{index, label, score, bbox_2d(0-1000 归一化), polygon, task_type}`。
- [base.py](file:///workspace/glmocr/layout/base.py) — `BaseLayoutDetector` ABC：`start()` / `stop()` / `process()` 三件套。
- [preprocess_pool/pool.py](file:///workspace/glmocr/preprocess_pool/pool.py) — 4 进程池 + `submit()` / `join()` 接口。
- [preprocess_pool/protocols.py](file:///workspace/glmocr/preprocess_pool/protocols.py) — `Region` + `DocumentPreprocessor` Protocol。
- [config.py](file:///workspace/glmocr/config.py) — `LayoutConfig` 已包含 `model_dir / device / threshold / label_task_mapping` 等所有布局检测需要的参数。

### 2.2 现有 `BaseLayoutDetector` 的复用点
- 我们的 `LayoutDetectorAdapter` 接受任何 `BaseLayoutDetector` 实例。
- 用户可以继续用 `PPDocLayoutDetector(model_dir=...)`，零改动。
- Adapter 负责把 `process()` 的输出转成 `Region` 列表（归一化坐标 → 像素坐标、cropped image、PIL Image 重建）。

### 2.3 用户使用流程的变化
**改前**（一次性实现一个巨类）：
```python
class MyDocPreprocessor:
    def process(self, image, **_):
        # 1) 文档检测、2) 方向、3) 扭曲、4) 布局 — 全部塞这
        ...
```

**改后**（4 个独立类 + 1 个 pipeline）：
```python
class MyDocDetector(BaseDocumentDetector):
    def _load_model(self):  return load_my_model(...)
    def _run_inference(self, x):  return self._model(x)

# 同理 MyOriDetector / MyDistCorrector

pipeline = DocumentPreprocessorPipeline(
    doc_detector=MyDocDetector(model_dir="...", device="cuda:0"),
    ori_detector=MyOriDetector(model_dir="..."),
    dist_corrector=MyDistCorrector(model_dir="..."),
    layout_detector=LayoutDetectorAdapter(PPDocLayoutDetector(cfg.pipeline.layout)),
)

with PreprocessPool(lambda: pipeline_factory(pipeline_cfg), config) as pool:
    pool.submit(...)
```

---

## 3. 拟新增文件 (Proposed Changes)

```
glmocr/preprocess_pool/
├── detectors/                      ← 新目录
│   ├── __init__.py                 ← 公开 API
│   ├── base.py                     ← Protocols / ABCs / 数据类
│   ├── document_detector.py        ← BaseDocumentDetector 骨架
│   ├── orientation_detector.py     ← BaseOrientationDetector 骨架
│   ├── distortion_corrector.py     ← BaseDistortionCorrector 骨架
│   └── layout_adapter.py           ← LayoutDetectorAdapter (复用 BaseLayoutDetector)
├── pipeline.py                     ← DocumentPreprocessorPipeline (orchestrator)
├── (既有) protocols.py / tasks.py / pool.py / async_ocr.py / aggregator.py / preprocess_worker.py / __init__.py
└── tests/
    ├── test_pipeline.py            ← 新增
    └── (既有) test_preprocess_pool.py / smoke_test.py
```

**不修改** 任何现有 SDK 文件（包括 `layout/`、`pipeline.py`、`config.py`），保证完全向后兼容。

---

### 3.1 `detectors/base.py` — 接口契约 + 数据类

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, List, Optional, Protocol, Tuple
import numpy as np
from PIL import Image


# ----------------------------- 数据类 -----------------------------

class Orientation(IntEnum):
    ROT_0   = 0
    ROT_90  = 90
    ROT_180 = 180
    ROT_270 = 270


@dataclass
class DocumentBox:
    """Stage-1 输出：原图中文档区域的位置与 4 角点。"""
    bbox: Tuple[int, int, int, int]            # (x1, y1, x2, y2), 原图像素
    corners: Optional[np.ndarray] = None       # shape (4, 2), 浮点像素
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


# ---------------------- 抽象基类 (用户子类化) ----------------------

class BaseDetector(ABC):
    """所有 4 个检测器的共同生命周期：init / start / stop / __call__。

    子类实现:
        _load_model() -> Any          # 加载自己的模型
        _run_inference(x) -> Any      # 模型推理

    继承的骨架提供:
        _preprocess(image) -> np.ndarray  # resize / 归一化
        _postprocess(raw) -> 输出          # NMS / 反归一化 / 解码
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cpu",
        input_size: Tuple[int, int] = (640, 640),
        **kwargs: Any,
    ):
        self.model_dir = model_dir
        self.device = device
        self.input_size = input_size
        self._model: Any = None
        self._started = False

    # ---- 用户必须重写 ----
    @abstractmethod
    def _load_model(self) -> Any:
        """从 model_dir 加载模型（ONNX / PyTorch / TensorRT 由你决定），
        放到 self.device，返回模型对象。"""
    @abstractmethod
    def _run_inference(self, preprocessed: np.ndarray) -> Any:
        """模型推理，preprocess 后的输入 → 模型原始输出。"""

    # ---- 骨架默认实现（古典 CV），用户可重写 ----
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """resize 到 self.input_size，RGB → float32 [0,1]，HWC → CHW。"""
        img = image.convert("RGB").resize(self.input_size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))  # CHW

    def _postprocess(self, raw: Any) -> Any:
        """默认 no-op。子类按模型输出解码。"""
        return raw

    # ---- 生命周期 ----
    def start(self) -> None:
        if self._started:
            return
        self._model = self._load_model()
        self._on_start()
        self._started = True

    def _on_start(self) -> None:
        """子类可重写做 warmup / 绑 device。"""
        pass

    def stop(self) -> None:
        if not self._started:
            return
        self._on_stop()
        self._model = None
        self._started = False

    def _on_stop(self) -> None:
        """子类可重写释放显存。"""
        pass

    def __call__(self, image: Image.Image) -> Any:
        if not self._started:
            raise RuntimeError(f"{type(self).__name__}.start() not called")
        x = self._preprocess(image)
        raw = self._run_inference(x)
        return self._postprocess(raw)

    # ---- Protocol 自检 ----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # 友好提示: 子类必须实现两个抽象方法
        if cls.__abstractmethods__:
            missing = ", ".join(sorted(cls.__abstractmethods__))
            # 不强制 fail；只在文档里提示
            pass
```

**为什么 `_preprocess` / `_postprocess` 默认是 CV 通用做法**：
- 大多数检测模型输入都是 CHW 归一化 float32，size 由用户配置。
- 后处理因模型而异（分类 argmax vs 检测 NMS），骨架只提供最通用的，子类重写。

---

### 3.2 `detectors/document_detector.py` — Stage 1

```python
from .base import BaseDetector, DocumentBox
import numpy as np
from PIL import Image
import cv2


class BaseDocumentDetector(BaseDetector):
    """Stage 1: 找到图像里的文档区域（一张照片里的"那块文档"）。

    _run_inference 的输出契约: dict 包含
        - 'mask': (H, W) np.ndarray float32, 文档区域概率图
        - 或 'boxes': (N, 4) np.ndarray + 'scores': (N,) + 'keypoints' 等
    骨架默认 _postprocess 处理 'mask'。
    """

    def __init__(self, mask_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.mask_threshold = mask_threshold

    def _postprocess(self, raw: dict) -> DocumentBox:
        """从 mask 找 4 角点 + bbox。"""
        if isinstance(raw, dict) and "mask" in raw:
            return self._mask_to_document_box(raw["mask"])
        raise NotImplementedError(
            f"{type(self).__name__}._postprocess expects dict with 'mask'; "
            f"override _postprocess for boxes-style outputs."
        )

    def _mask_to_document_box(self, mask: np.ndarray) -> DocumentBox:
        h, w = mask.shape
        bin_mask = (mask > self.mask_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return DocumentBox(bbox=(0, 0, w, h), corners=None, confidence=0.0)
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 4:
            return DocumentBox(bbox=(0, 0, w, h), corners=None, confidence=0.0)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            # 退化: fallback to minAreaRect
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
        else:
            box = approx.reshape(4, 2).astype(np.float32)
        # 排序为 top-left, top-right, bottom-right, bottom-left
        box = self._order_corners(box)
        x1, y1 = box.min(axis=0).astype(int)
        x2, y2 = box.max(axis=0).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return DocumentBox(bbox=(x1, y1, x2, y2), corners=box, confidence=1.0)

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """4 个点按 tl, tr, br, bl 顺序。"""
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[s.argmin()]
        br = pts[s.argmax()]
        tr = pts[d.argmin()]
        bl = pts[d.argmax()]
        return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)
```

**用户只需实现**（示意）：
```python
class MyDocDetector(BaseDocumentDetector):
    def _load_model(self):
        import onnxruntime as ort
        return ort.InferenceSession(f"{self.model_dir}/doc.onnx", providers=["CUDAExecutionProvider"])

    def _run_inference(self, x):
        return {"mask": self._model.run(None, {"input": x[None]})[0][0, 0]}
```

---

### 3.3 `detectors/orientation_detector.py` — Stage 2

```python
from .base import BaseDetector, Orientation
import numpy as np
from PIL import Image


class BaseOrientationDetector(BaseDetector):
    """Stage 2: 检测文档朝向（0/90/180/270）。

    _run_inference 的输出契约: shape (4,) 的 logits 数组或 (1, 4)。
    """

    def __init__(self, **kwargs):
        super().__init__(input_size=(224, 224), **kwargs)

    def _postprocess(self, raw) -> Orientation:
        logits = np.asarray(raw).squeeze()
        if logits.ndim > 1:
            logits = logits[0]
        idx = int(np.argmax(logits))
        return [Orientation.ROT_0, Orientation.ROT_90, Orientation.ROT_180, Orientation.ROT_270][idx]

    def __call__(self, image: Image.Image) -> Orientation:
        angle = super().__call__(image)
        # 角度以"使文档朝正"为准: 逆时针补正
        # 如果模型训练时 "0" 表示"文档已经朝正"，则 angle 就是要 rotate 的度数
        return angle
```

---

### 3.4 `detectors/distortion_corrector.py` — Stage 3

```python
from .base import BaseDetector
import numpy as np
import cv2
from PIL import Image


class BaseDistortionCorrector(BaseDetector):
    """Stage 3: 透视/扭曲矫正。

    输入: (cropped_image, corners_4x2)
    输出: straightened Image

    接口里 _run_inference 的契约: 返回 rectified 的 image array (H, W, 3) uint8，
    或返回新的 4 角点让我们用 cv2.warpPerspective。
    """

    def __call__(self, image: Image.Image, corners: np.ndarray | None = None) -> Image.Image:
        # 1) 调用模型 / 几何方法得到变换后图像
        if not self._started:
            raise RuntimeError(f"{type(self).__name__}.start() not called")
        x = self._preprocess(image)
        raw = self._run_inference(x)
        rectified = self._postprocess(raw, image, corners)
        return rectified

    def _postprocess(self, raw, image: Image.Image, corners: np.ndarray | None) -> Image.Image:
        """默认行为: 如果 raw 是 ndarray 当作矫正后图像；如果是 corner 数组就 warpPerspective。"""
        if isinstance(raw, np.ndarray) and raw.ndim == 3:
            return Image.fromarray(raw.astype(np.uint8))
        if isinstance(raw, np.ndarray) and raw.shape == (4, 2) and corners is not None:
            return self._warp(image, corners, raw)
        raise NotImplementedError(
            "Override _postprocess in your corrector; expected rectified image array or new corners."
        )

    @staticmethod
    def _warp(image: Image.Image, src_corners: np.ndarray, dst_corners: np.ndarray) -> Image.Image:
        """Compute homography from src→dst and warp."""
        M, _ = cv2.findHomography(src_corners, dst_corners, cv2.RANSAC)
        w, h = image.size
        warped = cv2.warpPerspective(np.array(image), M, (w, h))
        return Image.fromarray(warped)
```

---

### 3.5 `detectors/layout_adapter.py` — Stage 4（直接复用 SDK 现有 detector）

```python
from .base import BaseDetector
from PIL import Image
from typing import List
from glmocr.preprocess_pool.protocols import Region
from glmocr.layout.base import BaseLayoutDetector


class LayoutDetectorAdapter(BaseDetector):
    """Stage 4: 复用 SDK 的 BaseLayoutDetector (PP-DocLayoutV3 / 自定义)。

    注意: BaseLayoutDetector 的 process() 返回归一化坐标 [0, 1000]，adapter 负责
    转成像素并 crop 出 Region 的 image。
    """

    def __init__(self, base_detector: BaseLayoutDetector, **kwargs):
        # base_detector 自己有 start()/stop()，adapter 不重复管理生命周期
        self._detector = base_detector
        self._started = False
        # 忽略 model_dir/device/input_size

    def start(self) -> None:
        self._detector.start()
        self._started = True

    def stop(self) -> None:
        self._detector.stop()
        self._started = False

    def _load_model(self):
        return self._detector  # 实际模型在 self._detector 内

    def _run_inference(self, preprocessed):
        # 不走 numpy 路径 — 直接调 self._detector.process
        raise NotImplementedError("Use LayoutDetectorAdapter.__call__ directly")

    def __call__(self, image: Image.Image) -> List[Region]:
        if not self._started:
            raise RuntimeError("LayoutDetectorAdapter not started")
        all_results, _ = self._detector.process(
            images=[image], save_visualization=False, use_polygon=False
        )
        return self._convert(all_results[0], image)

    @staticmethod
    def _convert(detections: list, image: Image.Image) -> List[Region]:
        w, h = image.size
        out: List[Region] = []
        for d in detections:
            x1n, y1n, x2n, y2n = d["bbox_2d"]
            x1 = int(x1n * w / 1000); y1 = int(y1n * h / 1000)
            x2 = int(x2n * w / 1000); y2 = int(y2n * h / 1000)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))
            # 还原 polygon 到像素空间
            poly = [
                [int(px * w / 1000), int(py * h / 1000)]
                for px, py in d.get("polygon", [])
            ]
            out.append(Region(
                image=crop,
                bbox=(x1, y1, x2, y2),
                task_type=d["task_type"],
                label=d["label"],
                polygon=poly,
                metadata={"score": d["score"], "index": d["index"]},
            ))
        return out
```

**不抛 `NotImplementedError` 给用户** —— layout 是开箱即用的，零代码。

---

### 3.6 `pipeline.py` — `DocumentPreprocessorPipeline`

```python
from __future__ import annotations
import io
from typing import Any, List
from PIL import Image

from glmocr.preprocess_pool.detectors.document_detector import BaseDocumentDetector
from glmocr.preprocess_pool.detectors.orientation_detector import BaseOrientationDetector
from glmocr.preprocess_pool.detectors.distortion_corrector import BaseDistortionCorrector
from glmocr.preprocess_pool.detectors.layout_adapter import LayoutDetectorAdapter
from glmocr.preprocess_pool.protocols import DocumentPreprocessor, Region
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentPreprocessorPipeline(DocumentPreprocessor):
    """把 4 个检测器按 文档→方向→扭曲→布局 串成一个标准 pipeline。

    用法::

        pipeline = DocumentPreprocessorPipeline(
            doc_detector=MyDocDetector(model_dir="...", device="cuda:0"),
            ori_detector=MyOriDetector(model_dir="..."),
            dist_corrector=MyDistCorrector(model_dir="..."),
            layout_detector=LayoutDetectorAdapter(PPDocLayoutDetector(cfg.pipeline.layout)),
        )
        pipeline.start()
        regions = pipeline.process(image)   # List[Region]
        pipeline.stop()

    它自身实现 DocumentPreprocessor Protocol，可直接喂给 PreprocessPool::

        pool = PreprocessPool(
            preprocessor=lambda: make_pipeline(cfg),  # 每个 worker 各自 start
            config=cfg,
        )
    """

    def __init__(
        self,
        doc_detector: BaseDocumentDetector,
        ori_detector: BaseOrientationDetector,
        dist_corrector: BaseDistortionCorrector,
        layout_detector: LayoutDetectorAdapter,
    ):
        self._doc = doc_detector
        self._ori = ori_detector
        self._dist = dist_corrector
        self._lay = layout_detector
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._doc.start()
        self._ori.start()
        self._dist.start()
        self._lay.start()
        self._started = True
        logger.info("DocumentPreprocessorPipeline started")

    def stop(self) -> None:
        if not self._started:
            return
        for det in (self._doc, self._ori, self._dist, self._lay):
            try:
                det.stop()
            except Exception:
                logger.exception("Error stopping %s", type(det).__name__)
        self._started = False

    def process(self, image: Image.Image, **kwargs: Any) -> List[Region]:
        if not self._started:
            raise RuntimeError("DocumentPreprocessorPipeline.start() not called")

        # Stage 1: document detection
        box = self._doc(image)
        cropped = image.crop(box.bbox)

        # Stage 2: orientation
        angle = self._ori(cropped)
        if int(angle) != 0:
            rotated = cropped.rotate(-int(angle), expand=True)
        else:
            rotated = cropped

        # Stage 3: distortion correction
        # 注意: corners 来自 stage 1，是在 *原图* 坐标。旋转之后需要重新映射。
        # 简单处理: 只在 angle==0 时用原 corners；否则让 dist_corrector 自己从图里找。
        corners = box.corners if int(angle) == 0 else None
        if corners is not None:
            # 减去 bbox 偏移，映射到 cropped 坐标系
            from glmocr.preprocess_pool.detectors.document_detector import BaseDocumentDetector
            corners = corners - np.array([box.bbox[0], box.bbox[1]], dtype=np.float32)
        corrected = self._dist(rotated, corners=corners)

        # Stage 4: layout detection
        regions = self._lay(corrected)
        return regions
```

**注意**：`process()` 内的 `corners` 偏移计算需要 `import numpy as np` —— 实施时在文件顶部加。

---

### 3.7 `detectors/__init__.py` 暴露

```python
from .base import BaseDetector, DocumentBox, Orientation
from .document_detector import BaseDocumentDetector
from .orientation_detector import BaseOrientationDetector
from .distortion_corrector import BaseDistortionCorrector
from .layout_adapter import LayoutDetectorAdapter
```

并在 `glmocr/preprocess_pool/__init__.py` 增加：

```python
from .pipeline import DocumentPreprocessorPipeline
from .detectors import (
    BaseDocumentDetector, BaseOrientationDetector, BaseDistortionCorrector,
    LayoutDetectorAdapter, DocumentBox, Orientation,
)
```

---

## 4. 假设与决策 (Assumptions & Decisions)

| 假设 / 决策 | 说明 |
|------------|------|
| **A1**：3 个自定义检测器用 **ABC + NotImplementedError** 骨架；用户只需实现 `_load_model()` 和 `_run_inference()`。 | 双方确认的「骨架 + CV 后处理」方案。 |
| **A2**：layout 检测器走 `LayoutDetectorAdapter` 包装 `BaseLayoutDetector`，**直接复用** SDK 现有 `PPDocLayoutDetector`，**不重写**。 | 改 PP-DocLayoutV3 的代码风险大，且已经能跑；新代码只做协议转换。 |
| **A3**：检测器之间不共享 model，每个 worker 进程独立加载 4 个模型。 | 与 `PreprocessPool` 的 fork 模型一致。用户在 `preprocessor_factory` 里创建一次完整 pipeline，每个 worker 都跑这个 factory。 |
| **A4**：不写 config 集成（不往 `config.yaml` 加 detector 子段）。 | 模型路径因用户实现而异；用户在 Python 代码里传 `model_dir` 更灵活。Layout 仍复用现有 `pipeline.layout` 配置。 |
| **A5**：`Orientation` 语义为「模型预测 *文档已朝正时* 的旋转量」。预测 0 → 不旋转；预测 90 → 图被顺时针转了 90，要 `rotate(-90, expand=True)`。 | 文档里明确写出。 |
| **A6**：corners 跨阶段的坐标系由 pipeline 负责重映射；distortion corrector 的 API 是 `(image, corners=None)`，corners 为 None 时由 corrector 内部寻找。 | 让 corrector 自由发挥。 |
| **D1**：不改 `protocols.py` 的 `DocumentPreprocessor` Protocol。 | 保留完全自定义的入口。 |
| **D2**：不改 `BaseLayoutDetector`。 | 保持 SDK 完全向后兼容。 |
| **D3**：不引入新第三方包（numpy / PIL / cv2 已存在）。 | 极小依赖。 |

---

## 5. 验证步骤 (Verification)

1. **单测** `glmocr/preprocess_pool/tests/test_pipeline.py`：
   - mock 4 个检测器（返回固定 DocumentBox / Orientation / 简单 cropped image / 固定 Region 列表），
     验证 `DocumentPreprocessorPipeline.process()` 的端到端流转，包括 corners 偏移、angle 非 0 时旋转。
   - 验证 `BaseDocumentDetector._mask_to_document_box` 对正方形 mask 给合理 4 角点。
   - 验证 `LayoutDetectorAdapter._convert()` 归一化 → 像素 + crop 一致。

2. **集成** —— 现有 `test_preprocess_pool.py` 跑全 13 个用例（Pipeline 应能直接被 PreprocessPool 使用）：
   ```python
   pool = PreprocessPool(preprocessor=lambda: build_pipeline(), config=cfg)
   pool.submit(...)
   pool.join()
   ```

3. **layout_adapter 真实跑通**：
   - 用 `PPDocLayoutDetector` + 真实模型 + 一张图，验证 adapter 给出的 Region 数量和 bbox 与直接调 `detector.process()` 给的归一化坐标一致（用归一化 → 像素反推验证）。

4. **回归**：跑 `pytest glmocr/preprocess_pool/tests/ glmocr/tests/test_unit.py` 全量。

---

## 6. 实施顺序

1. `detectors/base.py`（数据类 + BaseDetector ABC + 生命周期）
2. `detectors/document_detector.py`（含古典 CV 角点提取）
3. `detectors/orientation_detector.py`
4. `detectors/distortion_corrector.py`
5. `detectors/layout_adapter.py`（包装 BaseLayoutDetector）
6. `detectors/__init__.py`
7. `pipeline.py`（DocumentPreprocessorPipeline orchestrator）
8. 更新 `glmocr/preprocess_pool/__init__.py` 暴露新 API
9. 单测 `test_pipeline.py`
10. 跑全量测试 + 回归
