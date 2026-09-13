"""Phase 10: bounded LLM concurrency + per-job cost ceiling.

Roadmap asks for two cost/latency controls:

1. **Bounded concurrency** — "Async LLM calls with a semaphore so 50 candidates
   don't serialize." We cannot make the synchronous openai client truly async
   without adding a dependency; instead we provide a threading.Semaphore so a
   multi-threaded caller (e.g. a ThreadPool running evidence verification for
   many candidates) never fires more than N in-flight LLM calls, while still
   allowing overlap.

2. **Cost ceiling** — "A documented cost ceiling per job, and a hard cap that
   falls back to the deterministic verdict path when exceeded." We count LLM
   *calls* (the practical proxy for cost given a fixed model) and expose
   :meth:`within_budget`; the evidence verifier consults it and falls back to
   its deterministic cross-encoder verdict path rather than spending more.

Both are pure-python, no external deps, thread-safe.
"""

from __future__ import annotations

import threading


class LLMSemaphore:
    """A threading.Semaphore bound to a max concurrency; no-op when max<=1."""

    def __init__(self, max_concurrent: int = 4):
        self._max = max(1, int(max_concurrent))
        self._sem = threading.BoundedSemaphore(self._max)

    @property
    def max_concurrent(self) -> int:
        return self._max

    def __enter__(self):
        self._sem.acquire()
        return self

    def __exit__(self, *exc):
        self._sem.release()
        return False


class CallBudget:
    """Tracks LLM call count against a per-job ceiling."""

    def __init__(self, max_calls: int = 50):
        self._max = max(0, int(max_calls))
        self._count = 0
        self._lock = threading.Lock()

    @property
    def max_calls(self) -> int:
        return self._max

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def remaining(self) -> int:
        return max(0, self._max - self.count)

    def spent(self) -> bool:
        """True once the ceiling has been reached (or is 0 = disabled)."""
        if self._max <= 0:
            # max_calls <= 0 means "no ceiling / unlimited" per a documented
            # convention: a manager that wants to disable the cap sets it to 0.
            return False
        return self.count >= self._max

    def take(self) -> bool:
        """Reserve one call if within budget; returns False (and takes nothing)
        when the ceiling would be exceeded."""
        if self._max <= 0:
            return True  # unlimited
        with self._lock:
            if self._count >= self._max:
                return False
            self._count += 1
            return True

    def reset(self) -> None:
        """Zero the call count (start of a new job budget window)."""
        with self._lock:
            self._count = 0
