"""Phase 10: on-disk content-addressed LLM cache tests.

Proves the two Phase 10 guarantees that matter for cost: (1) identical inputs
return the cached value with zero recompute, and (2) changing ANY input — or
the operation's version stamp — changes the key, so a stale response is never
served after a prompt/model change (the "What can go wrong" failure mode from
the roadmap).
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.cache import LLMCache


def _cache_dir():
    return tempfile.mkdtemp(prefix="llm_cache_test_")


def test_set_get_roundtrip():
    cache = LLMCache(_cache_dir())
    key = cache.key_for("v1", "prompt-a")
    cache.set(key, {"skills": ["python", "docker"]})
    assert cache.get(key) == {"skills": ["python", "docker"]}


def test_miss_returns_none():
    cache = LLMCache(_cache_dir())
    assert cache.get("never-written") is None


def test_key_is_content_addressed_and_stable():
    cache = LLMCache(_cache_dir())
    k1 = cache.key_for("v1", "the same prompt")
    k2 = cache.key_for("v1", "the same prompt")
    k3 = cache.key_for("v1", "a different prompt")
    assert k1 == k2          # deterministic
    assert k1 != k3          # content-addressed: different input -> different key
    # Keys must not be guessable sequence numbers.
    assert k1.isalnum() and len(k1) == 16


def test_version_stamp_changes_key():
    """A prompt/operation version bump must invalidate the cache entry."""
    cache = LLMCache(_cache_dir())
    old = cache.key_for("v1", "the same prompt")
    new = cache.key_for("v2", "the same prompt")
    assert old != new


def test_get_or_set_caches_producer_result():
    cache = LLMCache(_cache_dir())
    key = cache.key_for("v1", "prompt")
    calls = []

    def producer():
        calls.append(1)
        return {"result": ["a"]}

    value, hit = cache.get_or_set(key, producer)
    assert value == {"result": ["a"]}
    assert hit is False

    value2, hit2 = cache.get_or_set(key, producer)
    assert value2 == {"result": ["a"]}
    assert hit2 is True          # served from cache, producer not re-run
    assert len(calls) == 1       # producer ran exactly once


def test_corrupt_entry_treated_as_miss_not_crash():
    cache = LLMCache(_cache_dir())
    key = cache.key_for("v1", "prompt")
    path = cache._path(key)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{ not valid json !!!")
    assert cache.get(key) is None


def test_only_json_serializable_values_are_written():
    cache = LLMCache(_cache_dir())
    key = cache.key_for("v1", "prompt")
    cache.set(key, object())
    # A non-serializable value is silently skipped; no entry on disk.
    assert cache.get(key) is None


def test_persisted_across_instances():
    """The cache lives on disk: a second instance in the same dir sees entries."""
    d = _cache_dir()
    c1 = LLMCache(d)
    key = c1.key_for("v1", "shared prompt")
    c1.set(key, {"ok": True})

    c2 = LLMCache(d)
    assert c2.get(key) == {"ok": True}


if __name__ == "__main__":
    test_set_get_roundtrip()
    test_miss_returns_none()
    test_key_is_content_addressed_and_stable()
    test_version_stamp_changes_key()
    test_get_or_set_caches_producer_result()
    test_corrupt_entry_treated_as_miss_not_crash()
    test_only_json_serializable_values_are_written()
    test_persisted_across_instances()
    print("ALL CACHE TESTS PASSED")