"""Conftest for preprocess_pool tests.

Ensures ``multiprocessing`` uses ``fork`` so the dummy preprocessor class
defined at module level in the test files is inherited by the child
processes (rather than re-imported, which would lose the dummy).
"""

from __future__ import annotations

import multiprocessing


def pytest_configure(config):
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:  # pragma: no cover
        # start method was already set elsewhere; nothing to do.
        pass
