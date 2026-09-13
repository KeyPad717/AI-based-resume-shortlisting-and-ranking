"""Phase 10: per-job cost ceiling tests.

The roadmap requirement: "A documented cost ceiling per job, and a hard cap that
falls back to the deterministic verdict path when exceeded." These tests prove
both halves: ``CallBudget`` arithmetic, and — end-to-end through
``EvidenceVerifier`` — that an exhausted budget yields the deterministic
cross-encoder verdicts instead of an exception or an un-billed LLM call.
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.evidence import CitationValidator, EvidenceVerifier
from rag.llm_call import LLMDispatcher
from rag.llm_guard import CallBudget
from rag.schemas import EvidenceSpan, NormalizedRequirement


def _span(chunk_id, cand, text, ce_score=None):
    return EvidenceSpan(
        chunk_id=chunk_id, candidate_id=cand, text=text, cross_encoder_score=ce_score
    )


def _req(name):
    return NormalizedRequirement(skill_name=name, tier="required", source="llm_classifier")


class FakeRetriever:
    def __init__(self, spans_by_skill):
        self._spans = spans_by_skill

    def for_requirement(self, requirement, candidate_id, alt_labels=None):
        return list(self._spans.get(requirement.skill_name, []))


class OkLLM:
    """Returns wrapped-object JSON (dict whose value is the verdict array) — the
    shape gpt-4o-mini's json_object mode can legitimately produce."""

    def __init__(self, content):
        self._content = content
        self.calls = []
        self.chat = SimpleNamespace(completions=self._Completions(self))

    class _Completions:
        def __init__(self, owner):
            self._owner = owner

        def create(self, **kwargs):
            self._owner.calls.append(kwargs)
            msg = SimpleNamespace(content=self._owner._content)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


# ---------------------------------------------------------------------------
# CallBudget arithmetic
# ---------------------------------------------------------------------------

def test_budget_take_until_ceiling():
    b = CallBudget(3)
    assert b.take() and b.take() and b.take()
    assert not b.take()   # ceiling reached
    assert b.spent()
    assert b.remaining == 0
    assert b.count == 3


def test_budget_zero_means_unlimited():
    b = CallBudget(0)
    assert not b.spent()
    assert b.take()
    assert not b.spent()


def test_budget_reset_restores():
    b = CallBudget(2)
    b.take()
    b.take()
    assert b.spent()
    b.reset()
    assert not b.spent()
    assert b.remaining == 2


# ---------------------------------------------------------------------------
# End-to-end: evidence verifier, ceiling exhausted -> deterministic fallback
# ---------------------------------------------------------------------------

def test_evidence_budget_exhausted_falls_back_deterministically():
    retriever = FakeRetriever({
        "terraform": [_span("c1", "candA", "terraform iac", ce_score=0.7)],
        "docker": [_span("c2", "candA", "docker ci", ce_score=0.4)],
        "quantum": [_span("c3", "candA", "quantum qiskit", ce_score=0.1)],
    })
    llm = OkLLM('[{"skill_name":"terraform","verdict":"demonstrated","cited_chunk_ids":["c1"],"quote":"terraform"}]')

    # Ceiling of exactly 1 already consumed -> every further LLM call is blocked.
    budget = CallBudget(1)
    budget.take()
    dispatcher = LLMDispatcher(
        llm,
        cache=None,
        log=None,
        semaphore=None,
        budget=budget,
        cache_enabled=False,
    )
    verifier = EvidenceVerifier(retriever, llm, CitationValidator(), dispatcher=dispatcher)

    ev = verifier.verify(
        "candA",
        [_req("terraform"), _req("docker"), _req("quantum")],
    )
    by_name = {v.skill_name: v for v in ev.verdicts}
    # Deterministic CE-score verdicts, NOT the LLM's proposed ones.
    assert by_name["terraform"].verdict == "demonstrated"  # 0.7 > 0.6
    assert by_name["docker"].verdict == "claimed_only"     # 0.4 in (0.3, 0.6]
    assert by_name["quantum"].verdict == "absent"          # 0.1 <= 0.3
    assert ev.rejected_citations == 0
    assert len(llm.calls) == 0  # the LLM was never called after the ceiling


def test_evidence_within_budget_still_calls_llm_batched():
    retriever = FakeRetriever({
        "terraform": [_span("c1", "candA", "terraform iac", ce_score=0.7)],
        "docker": [_span("c2", "candA", "docker ci", ce_score=0.4)],
    })
    # Wrapped-object JSON ({"result": [...]}) — the shape the Phase 9 parser fix
    # accepts. Must survive routing through the Phase 10 dispatcher.
    llm = OkLLM(json.dumps({"result": [
        {"skill_name": "terraform", "verdict": "demonstrated",
         "cited_chunk_ids": ["c1"], "quote": "terraform iac"},
        {"skill_name": "docker", "verdict": "claimed_only",
         "cited_chunk_ids": [], "quote": None},
    ]}))
    dispatcher = LLMDispatcher(
        llm,
        cache=None,
        log=None,
        semaphore=None,
        budget=CallBudget(50),
        cache_enabled=False,
    )
    verifier = EvidenceVerifier(retriever, llm, CitationValidator(), dispatcher=dispatcher)

    ev = verifier.verify("candA", [_req("terraform"), _req("docker")])
    by_name = {v.skill_name: v for v in ev.verdicts}
    assert by_name["terraform"].verdict == "demonstrated"
    assert by_name["docker"].verdict == "claimed_only"
    assert len(llm.calls) == 1   # one batched call, within budget


if __name__ == "__main__":
    test_budget_take_until_ceiling()
    test_budget_zero_means_unlimited()
    test_budget_reset_restores()
    test_evidence_budget_exhausted_falls_back_deterministically()
    test_evidence_within_budget_still_calls_llm_batched()
    print("ALL COST CEILING TESTS PASSED")