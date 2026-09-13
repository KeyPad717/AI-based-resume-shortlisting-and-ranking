"""Phase 10: one shared helper for every LLM (chat.completions) call.

Centralizes the four Phase 10 concerns so the three LLM call sites
(call_llm_extraction, SkillNormalizer._classify_canonical, EvidenceVerifier
batch) do not each reimplement them:

  * **content-addressed caching** — identical prompts short-circuit on disk;
  * **structured JSONL logging** — prompt hash, model, latency, tokens, op;
  * **bounded concurrency** — a semaphore limits in-flight calls;
  * **per-job cost ceiling** — CallBudget lets callers fall back deterministically.

The function is synchronous (the project's LLM client is synchronous) but
semaphore-bounded, so a threaded caller overlaps safely.

``usage`` is extracted from the response when present (OpenAI-style
``response.usage.prompt_tokens`` / ``completion_tokens``); many fakes and even the
real OpenRouter gpt-4o-mini client expose it, but it is optional and never
required.
"""

from __future__ import annotations

import time

from rag.cache import LLMCache
from rag.llm_guard import CallBudget, LLMSemaphore
from rag.llm_log import LLMCallLog


class LLMDispatcher:
    """Callable wrapper around an LLM client's chat.completions.create."""

    def __init__(
        self,
        llm_client,
        *,
        cache=None,
        log=None,
        semaphore=None,
        budget=None,
        cache_enabled=True,
    ):
        self._client = llm_client
        self._cache = cache or LLMCache()
        self._log = log or LLMCallLog()
        self._sem = semaphore
        self._budget = budget
        self._cache_enabled = bool(cache_enabled)

    @classmethod
    def build(cls, llm_client, settings=None):
        """Construct a dispatcher from RagSettings (or the effective defaults).

        Shared by the three LLM call sites so cache/log/concurrency/cost wiring
        exists in exactly one place.
        """
        from rag.config import RagSettings

        settings = settings or RagSettings()
        return cls(
            llm_client,
            cache=LLMCache(settings.llm_cache_dir) if settings.llm_cache_enabled else None,
            log=LLMCallLog(settings.llm_log_path, enabled=settings.llm_log_enabled),
            semaphore=LLMSemaphore(settings.llm_max_concurrency),
            budget=CallBudget(settings.llm_max_calls_per_job),
            cache_enabled=settings.llm_cache_enabled,
        )

    # -- budget / concurrency surface ---------------------------------------

    @property
    def budget(self):
        return self._budget

    def reset_budget(self) -> None:
        """Reset the per-job cost ceiling (used as a new job starts)."""
        if self._budget is not None:
            self._budget.reset()

    def within_budget(self) -> bool:
        """False once the per-job ceiling is exhausted (or True for unlimited)."""
        if self._budget is None:
            return True
        return not self._budget.spent()

    def reserve(self) -> bool:
        """Attempt to reserve one LLM call against the budget; False if exhausted."""
        if self._budget is None:
            return True
        return self._budget.take()

    # -- the one shared call path -------------------------------------------

    def call(
        self,
        *,
        op: str,
        prompt: str,
        model: str,
        cache_version: str,
        cache_key_supplement: str = "",
    ):
        """Run one chat.completions call through cache + log + semaphore + budget.

        Returns the parsed JSON response (dict/list) on success, or ``None`` on
        failure. Cache is keyed on (cache_version + prompt + supplement) so a
        repeat call with identical inputs short-circuits to zero LLM spend.
        """
        if self._cache_enabled:
            key = self._cache.key_for(cache_version, prompt, extra=cache_key_supplement)
            hit = self._cache.get(key)
            if hit is not None:
                self._log.record(
                    op=op, model=model, hit=True, prompt_hash=LLMCallLog.prompt_hash(prompt)
                )
                return hit

        if not self.reserve():
            # Budget exhausted: callers treat a None return exactly like a total
            # LLM failure, so every call site falls back deterministically
            # (rule-based extraction / keyword classification / CE verdicts)
            # with zero further spend. The refusal is still logged for the cost
            # story.
            self._log.record(op=op, model=model, blocked="budget_exhausted")
            return None

        enter = None
        if self._sem is not None:
            enter = self._sem.__enter__()
        t0 = time.time()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            payload = self._parse(raw)
            latency = time.time() - t0
            usage = self._extract_usage(response)
            self._log.record(
                op=op, model=model, hit=False,
                prompt_hash=LLMCallLog.prompt_hash(prompt),
                latency_ms=round(latency * 1000.0, 2),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
            if payload is not None and self._cache_enabled:
                self._cache.set(key, payload)
            return payload
        except Exception:
            raise
        finally:
            if enter is not None:
                self._sem.__exit__(None, None, None)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _parse(raw: str):
        """Parse a JSON LLM body; raises ValueError when it is not valid JSON.

        Raising (rather than returning None) lets the caller's existing retry
        loops treat a malformed body like any other transient failure, while a
        None return from ``call()`` is reserved for the budget-exhausted case.
        """
        import json
        import re

        cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except ValueError:
            raise ValueError("LLM response was not parseable JSON") from None

    @staticmethod
    def _extract_usage(response) -> dict:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        return {"prompt_tokens": pt, "completion_tokens": ct}
