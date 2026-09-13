"""Phase 10: bounded LLM concurrency tests.

The roadmap's concern: "Async LLM calls with a semaphore so 50 candidates don't
serialize." Our client is synchronous, so the guarantee we can make is the
opposite half: with threaded callers, the number of in-flight LLM calls never
exceeds the configured ceiling. These tests prove both ``LLMSemaphore`` in
isolation and the dispatcher end-to-end.
"""

import json
import os
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.llm_call import LLMDispatcher
from rag.llm_guard import LLMSemaphore


def test_semaphore_never_exceeds_max_inflight():
    sem = LLMSemaphore(3)
    active = []
    active_lock = threading.Lock()
    peak = [0]

    def worker():
        with sem:
            with active_lock:
                active.append(1)
                peak[0] = max(peak[0], len(active))
            time.sleep(0.05)
            with active_lock:
                active.pop()

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] <= 3


class SlowCountingLLM:
    """Fake LLM with a small artificial latency so threads actually overlap."""

    def __init__(self, payload, delay=0.05):
        self._payload = payload
        self._delay = delay
        self.calls = []
        self.inflight = []
        self.peak = [0]
        self.inflight_lock = threading.Lock()
        self.chat = SimpleNamespace(completions=self._Completions(self))

    class _Completions:
        def __init__(self, owner):
            self._owner = owner

        def create(self, **kwargs):
            owner = self._owner
            owner.calls.append(kwargs)
            with owner.inflight_lock:
                owner.inflight.append(1)
                owner.peak[0] = max(owner.peak[0], len(owner.inflight))
            time.sleep(owner._delay)
            with owner.inflight_lock:
                owner.inflight.pop()
            msg = SimpleNamespace(content='{"ok": true}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def test_dispatcher_bounds_concurrent_calls():
    client = SlowCountingLLM({"ok": True}, delay=0.04)
    cache_dir = tempfile.mkdtemp()
    d = LLMDispatcher(
        client,
        cache=None,
        log=None,
        semaphore=LLMSemaphore(3),
        budget=None,
        cache_enabled=False,
    )
    results = []

    def worker(i):
        # distinct prompts so cache can never hide concurrency
        results.append(d.call(
            op="evidence", prompt="p%d" % i, model="m", cache_version="v1"
        ))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(client.calls) == 10
    assert client.peak[0] <= 3
    assert all(r == {"ok": True} for r in results)


def test_semaphore_max_one_serializes():
    sem = LLMSemaphore(1)
    active = [0]
    peak = [0]
    lock = threading.Lock()

    def worker():
        with sem:
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.03)
            with lock:
                active[0] -= 1

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


class EchoingLLM:
    """Fake LLM whose payload is deterministically derived from the prompt text.

    Echoing the prompt back means each of N distinct prompts yields a distinct
    result, so any accidental cross-talk between concurrent calls (one thread's
    result leaking into another's) surfaces as a mismatch instead of being
    masked by every call returning an identical canned value.
    """

    def __init__(self, delay=0.02):
        self._delay = delay
        self.calls = []
        self.chat = SimpleNamespace(completions=self._Completions(self))

    class _Completions:
        def __init__(self, owner):
            self._owner = owner

        def create(self, **kwargs):
            owner = self._owner
            prompt = kwargs["messages"][0]["content"]
            owner.calls.append(prompt)
            time.sleep(owner._delay)
            msg = SimpleNamespace(content=json.dumps({"echo": prompt}))
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def test_concurrency_does_not_change_output():
    prompts = ["prompt-%d" % i for i in range(6)]

    def run(batch, concurrency):
        dispatcher = LLMDispatcher(
            EchoingLLM(),
            cache=None,
            log=None,
            semaphore=LLMSemaphore(concurrency),
            budget=None,
            cache_enabled=False,  # both runs must genuinely execute, never cache-hit
        )
        results = {}
        results_lock = threading.Lock()

        def worker(prompt):
            out = dispatcher.call(op="evidence", prompt=prompt, model="m", cache_version="v1")
            with results_lock:
                results[prompt] = out

        threads = [threading.Thread(target=worker, args=(p,)) for p in batch]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results, dispatcher._client

    serial, serial_client = run(prompts, concurrency=1)
    parallel, parallel_client = run(prompts, concurrency=3)

    # identical key set, and identical results key for key
    assert set(serial) == set(parallel)
    for p in prompts:
        assert serial[p] == parallel[p]
    assert serial == parallel

    # both runs really executed every prompt (no cache short-circuit)
    assert sorted(serial_client.calls) == sorted(prompts)
    assert sorted(parallel_client.calls) == sorted(prompts)


if __name__ == "__main__":
    test_semaphore_never_exceeds_max_inflight()
    test_dispatcher_bounds_concurrent_calls()
    test_semaphore_max_one_serializes()
    print("ALL CONCURRENCY TESTS PASSED")