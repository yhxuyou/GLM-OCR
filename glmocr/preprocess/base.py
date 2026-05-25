"""Preprocessing base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import numpy as np


class BasePreprocessor(ABC):
    """Base class for GPU-accelerated image preprocessors.

    Defines a unified lifecycle interface. All preprocessor models are small
    and intended to be loaded once in :meth:`start`, kept resident on GPU,
    and released in :meth:`stop`.
    """

    def __init__(self, config: Any):
        """Initialize.

        Args:
            config: The corresponding preprocessor config object.
        """
        self.config = config
        self._model = None
        self._device = None

    @abstractmethod
    def start(self):
        """Load model to GPU / CPU and set up any pre-/post-processors.

        Called once at pipeline startup. After this returns the preprocessor
        is ready for repeated :meth:`process` calls from any thread.
        """

    @abstractmethod
    def stop(self):
        """Unload model and free GPU resources.

        Called once at pipeline shutdown.
        """

    @abstractmethod
    def process(
        self, image: "np.ndarray", **kwargs: Any
    ) -> "np.ndarray":
        """Run preprocessing on a single image.

        Args:
            image: Input image as a **numpy** array (H, W, C) in BGR or RGB
                   format.  The concrete implementation documents which format
                   it expects.
            **kwargs: Implementation-specific options.

        Returns:
            Processed image as a numpy array (H, W, C).
        """

    def is_ready(self) -> bool:
        """Return ``True`` when the model has been loaded via :meth:`start`."""
        return self._model is not None

    @property
    def device(self) -> Optional[str]:
        """Device the model is currently placed on (``"cpu"`` / ``"cuda:0"`` / ...)."""
        return self._device