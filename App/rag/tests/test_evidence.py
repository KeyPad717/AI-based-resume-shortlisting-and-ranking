import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.evidence import CitationValidator, EvidenceVerifier
from rag.prompts import build_evidence_prompt
from rag.schemas import EvidenceSpan, NormalizedRequirement, SkillVerdict


def _span(chunk_id, candidate_id, text, ce_score=None):
    return EvidenceSpan(
        chunk_id=chunk_id, candidate_id=candidate_id, text=text, cross_encoder_score=ce_score
    )


# --------------------------------------------------------------------------
# Fake infrastructure for EvidenceVerifier
# --------------------------------------------------------------------------

class FakeRetriever:
    """SkillEvidenceRetriever stand-in: returns configured spans per skill."""

    def __init__(self, spans_by_skill):
        # spans_by_skill: skill_name -> list[EvidenceSpan] (or [] for absent)
        self._spans = spans_by_skill
        self.for_requirement_calls = []

    def for_requirement(self, requirement, candidate_id, alt_labels=None):
        self.for_requirement_calls.append((requirement.skill_name, candidate_id, alt_labels))
        return list(self._spans.get(requirement.skill_name, []))


class FakeLLMOk:
    """Returns fixed JSON content on the single batched call."""

    def __init__(self, content):
        self._content = content
        self.calls = []

    class _Completions:
        def __init__(self, owner):
            self._owner = owner

        def create(self, **kwargs):
            self._owner.calls.append(kwargs)
            msg = SimpleNamespace(content=self._owner._content)
            choice = SimpleNamespace(message=msg)
            return SimpleNamespace(choices=[choice])

    def __init__(self, content):
        self._content = content
        self.calls = []
        self.chat = SimpleNamespace(completions=self._Completions(self))


class FailingLLMOk:
    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("intentional total LLM failure")

    def __init__(self):
        self.chat = SimpleNamespace(completions=self._Completions())


def _req(name):
    return NormalizedRequirement(skill_name=name, tier="required", source="llm_classifier")


# --------------------------------------------------------------------------
# CitationValidator tests
# --------------------------------------------------------------------------

def test_validate_chunk_id_not_in_spans_downgraded():
    validator = CitationValidator()
    retrieved = [_span("c1", "candA", "Built Kubernetes clusters with Helm")]
    verdicts = [
        {"skill_name": "kubernetes", "verdict": "demonstrated",
         "cited_chunk_ids": ["bad1"], "quote": "Built Kubernetes"}
    ]
    validated, rejected = validator.validate(verdicts, retrieved, "candA")
    assert rejected == 1
    assert validated[0].verdict == "claimed_only"
    assert validated[0].quote is None


def test_validate_paraphrased_quote_downgraded():
    validator = CitationValidator()
    retrieved = [_span("c1", "candA", "I built Kubernetes clusters with Helm for autoscaling")]
    # 'with container orchestration' is a paraphrase, NOT a verbatim substring.
    verdicts = [
        {"skill_name": "kubernetes", "verdict": "demonstrated",
         "cited_chunk_ids": ["c1"], "quote": "I built container orchestration clusters"}
    ]
    validated, rejected = validator.validate(verdicts, retrieved, "candA")
    assert rejected == 1
    assert validated[0].verdict == "claimed_only"
    assert validated[0].quote is None


def test_validate_exact_quote_normalized_passes():
    validator = CitationValidator()
    retrieved = [_span("c1", "candA", "Deployed Docker containers to AWS ECS service.")]
    # Whitespace/case normalized still matches verbatim after collapse+lower.
    verdicts = [
        {"skill_name": "docker", "verdict": "demonstrated",
         "cited_chunk_ids": ["c1"], "quote": "deployed   docker containers to aws"}
    ]
    validated, rejected = validator.validate(verdicts, retrieved, "candA")
    assert rejected == 0
    assert validated[0].verdict == "demonstrated"
    assert validated[0].quote == "deployed   docker containers to aws"


# --------------------------------------------------------------------------
# EvidenceVerifier tests
# --------------------------------------------------------------------------

def test_verify_zero_span_requirement_absent_no_llm_for_it():
    spans_for_skill1 = [_span("c1", "candA", "Deep familiarity with Terraform modules.")]
    retriever = FakeRetriever({"terraform": spans_for_skill1})  # skill2 has NO spans
    llm = FakeLLMOk('[{"skill_name":"terraform","verdict":"demonstrated",'
                    '"cited_chunk_ids":["c1"],"quote":"Deep familiarity with Terraform"}]')
    verifier = EvidenceVerifier(retriever, llm, CitationValidator())

    ev = verifier.verify("candA", [_req("terraform"), _req("kubernetes")])

    # kubernetes had zero spans -> absent, and never sent to the LLM.
    kube = [v for v in ev.verdicts if v.skill_name == "kubernetes"][0]
    assert kube.verdict == "absent"
    assert kube.cited_chunk_ids == []
    # Exactly ONE batched LLM call happened (for terraform only).
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "terraform" in prompt
    assert "kubernetes" not in prompt  # LLM never saw the absent skill


def test_verify_total_llm_failure_uses_deterministic_fallback():
    retriever = FakeRetriever({
        "terraform": [_span("c1", "candA", "terraform iac", ce_score=0.7)],
        "docker": [_span("c2", "candA", "docker ci", ce_score=0.4)],
        "quantum": [_span("c3", "candA", "quantum qiskit", ce_score=0.1)],
        "kubernetes": [],  # zero-span stays absent
    })
    llm = FailingLLMOk()  # total failure -> deterministic fallback
    verifier = EvidenceVerifier(retriever, llm, CitationValidator())

    ev = verifier.verify(
        "candA",
        [_req("terraform"), _req("docker"), _req("quantum"), _req("kubernetes")],
    )
    # No exception; verdicts derived from cross_encoder_score.
    by_name = {v.skill_name: v for v in ev.verdicts}
    assert by_name["terraform"].verdict == "demonstrated"  # 0.7 > 0.6
    assert by_name["docker"].verdict == "claimed_only"     # 0.4 in (0.3, 0.6]
    assert by_name["quantum"].verdict == "absent"          # 0.1 <= 0.3
    assert by_name["kubernetes"].verdict == "absent"       # zero-span
    assert ev.rejected_citations == 0


def test_verify_llm_verdicts_run_through_validator():
    retriever = FakeRetriever({
        "terraform": [_span("c1", "candA", "Deep familiarity with Terraform modules.")],
        "docker": [_span("c2", "candA", "Run Docker in CI pipelines.")],
    })
    # LLM proposes a bad citation (quote not verbatim) for one skill -> code rejects it.
    llm = FakeLLMOk('[{"skill_name":"terraform","verdict":"demonstrated",'
                    '"cited_chunk_ids":["c1"],"quote":"familiarity with iac"},'
                    '{"skill_name":"docker","verdict":"demonstrated",'
                    '"cited_chunk_ids":["c2"],"quote":"Run Docker in CI"}]')
    verifier = EvidenceVerifier(retriever, llm, CitationValidator())
    ev = verifier.verify("candA", [_req("terraform"), _req("docker")])
    by_name = {v.skill_name: v for v in ev.verdicts}
    assert by_name["terraform"].verdict == "claimed_only"  # bad quote rejected
    assert by_name["terraform"].quote is None
    assert by_name["docker"].verdict == "demonstrated"     # good citation kept
    assert ev.rejected_citations == 1


def test_prompt_is_deterministic_snapshot():
    r = _req("kubernetes")
    spans = [_span("c1", "candA", "Managed Kubernetes clusters in production")]
    p1 = build_evidence_prompt("candA", [(r, spans)])
    p2 = build_evidence_prompt("candA", [(r, spans)])
    assert p1 == p2
    assert "PROMPT VERSION: v1" in p1


if __name__ == "__main__":
    test_validate_chunk_id_not_in_spans_downgraded()
    test_validate_paraphrased_quote_downgraded()
    test_validate_exact_quote_normalized_passes()
    test_verify_zero_span_requirement_absent_no_llm_for_it()
    test_verify_total_llm_failure_uses_deterministic_fallback()
    test_verify_llm_verdicts_run_through_validator()
    test_prompt_is_deterministic_snapshot()
    print("ALL EVIDENCE TESTS PASSED")
