"""Phase 10: on-disk content-addressed LLM response cache.

Motivation (roadmap §Phase 10): LLM extraction/classification/verdict calls are
the dominant wall-clock AND dollar cost in a re-score. Re-scoring the same job
should touch zero LLM endpoints. This cache stores LLM responses on disk keyed
by *what the call would output* (i.e. the exact prompt content + a version
stamp), so a second identical call short-circuits to the cached JSON.

Key discipline (see roadmap "What can go wrong"): every input that affects the
output must be part of the key — that is why each key is built from the prompt
*and* an explicit version stamp per operation. This mirrors the project's
existing version-stamp discipline (chunker_version, EVIDENCE_VERDICT_PROMPT_VERSION,
embed_model). If a prompt or model changes, the version stamp changes, the key
changes, and stale responses are never served.

No external cache dependencies: this is a plain JSON-per-key file cache (one
file per key) rather than Redis/Memcached, consistent with the project's
in-memory-first / no-external-persistence pattern.

OPTIMIZATION NOTE (roadmap §Phase 10: "Qdrant scalar quantization if memory
matters; ONNX or optimum int8 for the cross-encoder if latency does. Measure
before reaching for either."): quantization/ONNX are NOT justified here — no
measured pressure exists. The latency bottleneck after caching is the batched
LLM verdict call (one per candidate), not the cross-encoder or the bi-encoder:
the encoding+rerank path is already batched and sub-second per candidate (Phase
3/6), while a re-score of an unchanged job now makes ZERO model calls thanks to
the content-addressed cache. Scalar quantization or int8 ONNX would therefore
optimize already-latency-invisible code at the cost of deployment complexity;
revisit only if Qdrant is adopted in production and memory actually becomes a
measured constraint.
"""

from __future__ import annotations

import json
import os
import threading

from rag.hashing import content_id


class LLMCache:
    """Thread-safe, on-disk, content-addressed JSON cache for LLM responses.

    Each entry is a single JSON file named ``<key>.json`` under ``cache_dir``.
    The key is a truncated sha256 of (operation_version + serialized prompt),
    produced by :meth:`key_for`. Values are arbitrary JSON-serializable objects
    (the parsed LLM response).
    """

    def __init__(self, cache_dir: str | None = None, key_length: int = 16):
        self._dir = cache_dir or os.environ.get(
            "RAG_CACHE_DIR", os.path.join(".rag_cache", "llm")
        )
        self._key_length = key_length
        self._lock = threading.Lock()

    # -- key construction ---------------------------------------------------

    def key_for(self, version: str, *prompt_parts: str, extra: str = "") -> str:
        """Content-addressed cache key for a set of prompt inputs.

        ``version`` is the operation's version stamp (e.g. a prompt version or a
        model name); ``prompt_parts`` are the exact inputs that determine the
        output (prompt text, candidate id, skill names, serialized spans...).
        Changing any input — or the version stamp — changes the key, so a stale
        cached response is never served after a prompt/model change.
        """
        payload = "\x1f".join([version] + list(prompt_parts) + ([extra] if extra else []))
        return content_id(payload, self._key_length)

    # -- storage ------------------------------------------------------------

    def _path(self, key: str) -> str:
        return os.path.join(self._dir, key + ".json")

    def _ensure_dir(self) -> None:
        os.makedirs(self._dir, exist_ok=True)

    def get(self, key: str):
        """Return the cached parsed JSON value for ``key``, or None on miss/corrupt."""
        path = self._path(key)
        try:
            with self._lock:
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            # Unreadable/corrupt entries are treated as a miss, never a crash.
            return None

    def set(self, key: str, value) -> None:
        """Persist ``value`` (JSON-serializable) under ``key``."""
        try:
            payload = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return
        path = self._path(key)
        tmp = path + ".tmp"
        try:
            with self._lock:
                self._ensure_dir()
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                os.replace(tmp, path)
        except OSError:
            # Fail to write (e.g. read-only FS) never breaks the caller.
            self._cleanup_tmp(tmp)

    def get_or_set(self, key: str, producer):
        """Return cached value for ``key``, else compute via ``producer`` and cache it.

        ``producer`` is a zero-arg callable returning a JSON-serializable value.
        """
        hit = self.get(key)
        if hit is not None:
            return hit, True
        value = producer()
        if value is not None:
            self.set(key, value)
        return value, False

    def contains(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def clear(self) -> None:
        """Remove every cached entry (used in tests; not invoked in production flow)."""
        with self._lock:
            for name in os.listdir(self._dir):
                if name.endswith(".json"):
                    try:
                        os.remove(os.path.join(self._dir, name))
                    except OSError:
                        pass

    def size(self) -> int:
        with self._lock:
            try:
                return len([n for n in os.listdir(self._dir) if n.endswith(".json")])
            except OSError:
                return 0

    @staticmethod
    def _cleanup_tmp(tmp: str) -> None:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
