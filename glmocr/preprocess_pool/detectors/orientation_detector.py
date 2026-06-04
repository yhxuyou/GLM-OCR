"""Stage 2 detector: document orientation classification.

Subclass :class:`BaseOrientationDetector` and implement the two
abstract methods.  The default postprocessing expects a 4-class logits
array and returns an :class:`Orientation` enum value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .base import BaseDetector, Orientation


class BaseOrientationDetector(BaseDetector):
    """Abstract base for 4-way orientation classifiers.

    Example subclass::

        import torch
        from torchvision import models

        class MyOriDetector(BaseOrientationDetector):
            def _load_model(self):
                m = models.resnet18(num_classes=4)
                sd = torch.load(f"{self.model_dir}/ori.pt", map_location=self.device)
                m.load_state_dict(sd)
                m.eval().to(self.device)
                return m

            def _run_inference(self, x):
                import torch
                with torch.no_grad():
                    return self._model(torch.from_numpy(x)[None]).cpu().numpy()
    """

    # 4-class classification models typically use 224×224 inputs.
    DEFAULT_INPUT_SIZE = (224, 224)

    def __init__(self, **kwargs: Any):
        # Allow callers to override input_size; otherwise fall back to 224x224.
        kwargs.setdefault("input_size", self.DEFAULT_INPUT_SIZE)
        super().__init__(**kwargs)
        # Index → Orientation lookup; also defined as a class attr so
        # subclasses can override the label ordering if their model
        # was trained with a different convention.
        self._idx_to_orientation = (
            Orientation.ROT_0,
            Orientation.ROT_90,
            Orientation.ROT_180,
            Orientation.ROT_270,
        )

    def _postprocess(self, raw: Any) -> Orientation:
        """Pick the argmax over the 4-class logits and map to Orientation.

        Accepts shapes ``(4,)``, ``(1, 4)`` or ``(N, 4)`` (in which case
        only the first row is used).
        """
        logits = np.asarray(raw)
        if logits.ndim == 2:
            if logits.shape[0] != 1:
                raise ValueError(
                    f"Expected batch size 1 for orientation logits, got {logits.shape[0]}"
                )
            logits = logits[0]
        if logits.shape != (4,):
            raise ValueError(
                f"Orientation detector expects 4-class logits, got shape {logits.shape}"
            )
        idx = int(np.argmax(logits))
        if not 0 <= idx < len(self._idx_to_orientation):
            raise RuntimeError(
                f"argmax index {idx} is out of range for orientation lookup"
            )
        return self._idx_to_orientation[idx]


__all__ = ["BaseOrientationDetector"]
