"""Circuit breaker implementation for OCR service calls."""
from __future__ import annotations

import time
import threading
from enum import Enum
from typing import Callable, Optional, Any


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker for OCR service calls.

    Prevents cascading failures when the downstream OCR service is degraded.

    State machine:
        CLOSED   → normal operation, requests pass through
        OPEN     → failures exceed threshold, requests are rejected immediately
        HALF_OPEN → after reset_timeout, one probe request is allowed
    """

    def __init__(
        self,
        name: str = "ocr",
        fail_max: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        exclude_exceptions: Optional[tuple] = None,
    ):
        self.name = name
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.exclude_exceptions = exclude_exceptions or ()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self._success_count = 0
        self._total_calls = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute *func* through the circuit breaker.

        Args:
            func: The callable to execute.
            *args, **kwargs: Passed through to *func*.

        Returns:
            The return value of *func*.

        Raises:
            CircuitBreakerOpenError: If the circuit is OPEN.
            Any exception raised by *func* (except excluded ones).
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(self.name)

        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(self.name)
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.exclude_exceptions:
            raise
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            self._success_count += 1
            self._total_calls += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0

    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._total_calls += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.fail_max:
                self._state = CircuitState.OPEN

    def reset(self):
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0
            self._half_open_calls = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "total_calls": self._total_calls,
                "success_count": self._success_count,
            }


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is OPEN and rejects a call."""

    def __init__(self, name: str):
        super().__init__(f"Circuit breaker '{name}' is OPEN")
        self.breaker_name = name


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, max_rate: float = 50.0, time_window: float = 1.0):
        self.max_rate = max_rate
        self.time_window = time_window
        self._tokens = max_rate
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, block: bool = True) -> bool:
        """Acquire *tokens* from the bucket.

        Args:
            tokens: Number of tokens to acquire.
            block: If True, blocks until tokens are available.

        Returns:
            True if tokens were acquired, False otherwise.
        """
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

            if not block:
                return False
            time.sleep(0.01)

    def _refill(self):
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.max_rate,
            self._tokens + elapsed * (self.max_rate / self.time_window),
        )
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens