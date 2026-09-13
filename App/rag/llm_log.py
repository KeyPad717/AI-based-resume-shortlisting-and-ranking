"""Phase 10: structured JSONL logging of every LLM call.

Records one JSON line per LLM (chat.completions) call: prompt hash, model,
latency, token usage (when the client exposes it), the operation type, and for
evidence calls the citation-rejection count. This is the cost story AND the
debugging story the roadmap asks for (`structured logging of every LLM call to
JSONL`).

Writing is append-only JSONL, batched to the OS via a line-buffered file held
for the process lifetime. Log writes are best-effort: a failure to log never
breaks an LLM call. Thread-safe via a lock.
"""

from __future__ import annotations

import json
import os
import threading
import time

from rag.hashing import content_id


class LLMCallLog:
    """Line-buffered JSONL sink for LLM call records."""

    def __init__(self, path: str | None = None, enabled: bool = True):
        self._path = path or os.environ.get(
            "RAG_LLM_LOG", os.path.join(".rag_logs", "llm_calls.jsonl")
        )
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._fh = None
        if self._enabled:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)

    def _open(self):
        if self._fh is None and self._enabled:
            self._fh = open(self._path, "a", encoding="utf-8")
        return self._fh

    def record(self, **fields) -> None:
        """Append one JSONL line. Best-effort — never raises into the caller.

        ``fields`` should be a flat dict; ``ts`` and ``prompt_hash`` are added
        when absent.
        """
        if not self._enabled:
            return
        fields.setdefault("ts", time.time())
        if "prompt_hash" in fields and fields["prompt_hash"] is None:
            fields.pop("prompt_hash")
        try:
            with self._lock:
                fh = self._open()
                if fh is None:
                    return
                fh.write(json.dumps(fields, ensure_ascii=False) + "\n")
                fh.flush()
        except OSError:
            pass

    @staticmethod
    def prompt_hash(prompt: str) -> str:
        return content_id(prompt, 12)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
