"""Phase 10: LLMDispatcher (cache + JSONL log + budget) unit tests.

Covers the integration guarantees: an identical prompt short-circuits to the
on-disk cache with zero underlying-client calls; the JSONL call log records
prompt hash / model / latency; a budget-exhausted call returns None so call
sites fall back deterministically.
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.cache import LLMCache
from rag.llm_call import LLMDispatcher
from rag.llm_guard import CallBudget, LLMSemaphore
from rag.llm_log import LLMCallLog


class CountingLLM:
    """OpenAI-shaped fake: returns fixed JSON on every call, counts calls."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = []
        self.chat = SimpleNamespace(completions=self._Completions(self))

    class _Completions:
        def __init__(self, owner):
            self._owner = owner

        def create(self, **kwargs):
            self._owner.calls.append(kwargs)
            msg = SimpleNamespace(content=json.dumps(self._owner._payload))
            choice = SimpleNamespace(message=msg)
            usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
            return SimpleNamespace(choices=[choice], usage=usage)


class FailingJSONLLM:
    """Fake whose body is not valid JSON -> dispatcher must raise ValueError."""

    class _Completions:
        def create(self, **kwargs):
            msg = SimpleNamespace(content="totally not json")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)

    def __init__(self):
        self.chat = SimpleNamespace(completions=self._Completions())


def _tmp_budget(max_calls=100):
    return CallBudget(max_calls)


def _dispatcher(llm, cache_dir, max_calls=100, sem_max=4):
    return LLMDispatcher(
        llm,
        cache=LLMCache(cache_dir),
        log=LLMCallLog(os.path.join(cache_dir, "calls.jsonl")),
        semaphore=LLMSemaphore(sem_max),
        budget=_tmp_budget(max_calls),
        cache_enabled=True,
    )


def test_identical_prompt_hits_cache_zero_client_calls():
    d = _dispatcher(CountingLLM({"skills": ["a"]}), tempfile.mkdtemp())
    r1 = d.call(op="extraction", prompt="hello", model="m", cache_version="v1")
    r2 = d.call(op="extraction", prompt="hello", model="m", cache_version="v1")
    assert r1 == {"skills": ["a"]}
    assert r2 == {"skills": ["a"]}
    assert d._client.calls  # one or more underlying calls
    assert d._client.calls[0]["messages"][0]["content"] == "hello"
    # The SECOND call must not have hit the underlying client again if the
    # first already persisted the cache entry. Since caching writes on first
    # call, a subsequent identical call is served from disk.
    assert len(d._client.calls) == 1


def test_changed_version_changes_cache_key_and_calls_again():
    client = CountingLLM({"v": 1})
    d = _dispatcher(client, tempfile.mkdtemp())
    d.call(op="extraction", prompt="same", model="m", cache_version="v1")
    d.call(op="extraction", prompt="same", model="m", cache_version="v2")
    # A prompt change must not be served from the old v1 cache entry.
    assert len(client.calls) == 2


def test_changed_prompt_calls_again():
    client = CountingLLM({"v": 1})
    d = _dispatcher(client, tempfile.mkdtemp())
    d.call(op="extraction", prompt="prompt-A", model="m", cache_version="v1")
    d.call(op="extraction", prompt="prompt-B", model="m", cache_version="v1")
    assert len(client.calls) == 2


def test_cache_disabled_always_calls_client():
    client = CountingLLM({"v": 1})
    cache_dir = tempfile.mkdtemp()
    d = LLMDispatcher(
        client,
        cache=LLMCache(cache_dir),
        log=LLMCallLog(os.path.join(cache_dir, "calls.jsonl")),
        semaphore=LLMSemaphore(2),
        budget=_tmp_budget(100),
        cache_enabled=False,
    )
    d.call(op="extraction", prompt="same", model="m", cache_version="v1")
    d.call(op="extraction", prompt="same", model="m", cache_version="v1")
    assert len(client.calls) == 2


def test_jsonl_log_records_call_and_hit_lines():
    cache_dir = tempfile.mkdtemp()
    log = LLMCallLog(os.path.join(cache_dir, "calls.jsonl"))
    d = LLMDispatcher(
        CountingLLM({"ok": 1}),
        cache=LLMCache(cache_dir),
        log=log,
        semaphore=LLMSemaphore(2),
        budget=_tmp_budget(100),
        cache_enabled=True,
    )
    d.call(op="evidence", prompt="p1", model="openai/gpt-4o-mini", cache_version="v1")
    d.call(op="evidence", prompt="p1", model="openai/gpt-4o-mini", cache_version="v1")
    log.close()

    lines = [json.loads(l) for l in open(os.path.join(cache_dir, "calls.jsonl")) if l.strip()]
    assert any(not rec.get("hit") for rec in lines)
    assert any(rec.get("hit") for rec in lines)
    first = lines[0]
    assert first["op"] == "evidence"
    assert first["model"] == "openai/gpt-4o-mini"
    assert first["prompt_hash"] and len(first["prompt_hash"]) == 12
    assert first["prompt_tokens"] == 10 and first["completion_tokens"] == 5
    assert "latency_ms" in first


def test_non_json_body_raises_value_error():
    d = _dispatcher(FailingJSONLLM(), tempfile.mkdtemp())
    try:
        d.call(op="evidence", prompt="x", model="m", cache_version="v1")
    except ValueError as e:
        assert "not parseable JSON" in str(e)
        return
    raise AssertionError("expected ValueError for non-JSON LLM body")


def test_budget_exhaustion_returns_none_and_logs_blocked():
    cache_dir = tempfile.mkdtemp()
    log = LLMCallLog(os.path.join(cache_dir, "calls.jsonl"))
    client = CountingLLM({"ok": 1})
    d = LLMDispatcher(
        client,
        cache=LLMCache(cache_dir),
        log=log,
        semaphore=LLMSemaphore(2),
        budget=CallBudget(1),  # ceiling: exactly one call
        cache_enabled=True,
    )
    r1 = d.call(op="evidence", prompt="p-first", model="m", cache_version="v1")
    # A DIFFERENT prompt misses the cache, so it must hit the budget ceiling.
    r2 = d.call(op="evidence", prompt="p-second", model="m", cache_version="v1")
    assert r1 == {"ok": 1}
    assert r2 is None  # budget exhausted -> caller's deterministic fallback
    assert len(client.calls) == 1
    log.close()
    lines = [json.loads(l) for l in open(os.path.join(cache_dir, "calls.jsonl")) if l.strip()]
    assert any(rec.get("blocked") == "budget_exhausted" for rec in lines)


def test_reset_budget_restores_spend_capacity():
    client = CountingLLM({"ok": 1})
    d = LLMDispatcher(
        client,
        cache=LLMCache(tempfile.mkdtemp()),
        log=LLMCallLog(os.path.join(tempfile.mkdtemp(), "c.jsonl")),
        semaphore=LLMSemaphore(2),
        budget=CallBudget(1),
        cache_enabled=True,
    )
    assert d.call(op="evidence", prompt="d0", model="m", cache_version="v1") == {"ok": 1}
    assert d.call(op="evidence", prompt="d1", model="m", cache_version="v1") is None
    d.reset_budget()
    assert d.call(op="evidence", prompt="d2", model="m", cache_version="v1") == {"ok": 1}


if __name__ == "__main__":
    test_identical_prompt_hits_cache_zero_client_calls()
    test_changed_version_changes_cache_key_and_calls_again()
    test_changed_prompt_calls_again()
    test_cache_disabled_always_calls_client()
    test_jsonl_log_records_call_and_hit_lines()
    test_non_json_body_raises_value_error()
    test_budget_exhaustion_returns_none_and_logs_blocked()
    test_reset_budget_restores_spend_capacity()
    print("ALL LLM CALL TESTS PASSED")