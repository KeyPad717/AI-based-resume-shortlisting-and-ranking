import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.ontology import ONTOLOGY_MATCH_FLOOR, SkillNormalizer


# --------------------------------------------------------------------------
# Lightweight fakes
# --------------------------------------------------------------------------

class FakeEmbedderWithSkill:
    """encode(texts) -> vector whose 2nd element is the skill string, so the
    fake ontology store can recover which skill was queried."""

    def encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 8
            v[0] = 0.5
            v[1] = t
            out.append(v)
        return out


class FakeOntologyStore:
    """Mimics VectorStore.search_dense over the esco collection, keyed by the
    skill string carried in the 2nd slot of the (otherwise dummy) query vector."""

    def __init__(self, lookup):
        # lookup: skill -> (score, preferred_label, alt_labels) or None
        self._lookup = lookup

    def search_dense(self, collection, query_vector, k, filters=None):
        info = self._lookup.get(query_vector[1])
        if info is None:
            return []
        score, preferred, alts = info
        payload = {"preferred_label": preferred, "alt_labels": alts, "concept_uri": "uri"}
        return [{"id": "concept-1", "score": score, "payload": payload}]


def make_store(lookup):
    return FakeOntologyStore(lookup)


class FakeLLMClient:
    """openai-like client returning a fixed JSON content string."""

    def __init__(self, content):
        self._content = content
        self.create_calls = []
        self.chat = SimpleNamespace(completions=self._Completions(self))

    class _Completions:
        def __init__(self, owner):
            self._owner = owner

        def create(self, **kwargs):
            self._owner.create_calls.append(kwargs)
            msg = SimpleNamespace(content=self._owner._content)
            choice = SimpleNamespace(message=msg)
            return SimpleNamespace(choices=[choice])


class FailingLLMClient:
    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("intentional LLM failure")

    def __init__(self):
        self.chat = SimpleNamespace(completions=self._Completions())


def _k8s_store():
    # "kubernetes" -> strong match (Kubernetes concept with alt labels);
    # anything else -> no match.
    return make_store(
        {
            "kubernetes": (0.80, "Kubernetes", ["container orchestration", "k8s"]),
            "k8s": (0.82, "Kubernetes", ["container orchestration", "k8s"]),
            "docker": (0.75, "Docker", ["containerization"]),
            "python": (0.78, "Python", ["pandas", "numpy"]),
        }
    )


# --------------------------------------------------------------------------

def test_normalize_success_llm_classifier():
    store = _k8s_store()
    normalizer = SkillNormalizer(store, FakeEmbedderWithSkill(),
                                 FakeLLMClient(json.dumps({
                                     "required": ["Kubernetes", "Python"],
                                     "preferred": ["Docker"],
                                 })))
    reqs = normalizer.normalize("JD text", ["k8s", "docker", "python"])
    assert len(reqs) == 3
    # Original strings preserved 1:1, source from LLM success.
    assert [r.skill_name for r in reqs] == ["k8s", "docker", "python"]
    assert all(r.source == "llm_classifier" for r in reqs)
    by_name = {r.skill_name: r for r in reqs}
    assert by_name["k8s"].tier == "required"      # canonical Kubernetes in required
    assert by_name["python"].tier == "required"   # canonical Python in required
    assert by_name["docker"].tier == "preferred"  # canonical Docker in preferred


def test_normalize_llm_failure_falls_back_keyword():
    import pipeline  # real classify_requirement_type lives here

    store = _k8s_store()
    normalizer = SkillNormalizer(store, FakeEmbedderWithSkill(), FailingLLMClient())

    failing_global_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingLLMClient._Completions())
    )
    with patch.object(pipeline, "client", failing_global_client):
        # falls to the real classify_requirement_type (global client made to
        # fail deterministically), then the keyword heuristic.
        reqs = normalizer.normalize("must have kubernetes python", ["kubernetes", "docker"])

    assert len(reqs) == 2
    assert all(r.source == "keyword_fallback" for r in reqs)


def test_both_tiers_fail_reuses_split_jd_sections():
    import pipeline
    from pipeline import split_jd_sections  # real canonical fallback path

    jd_text = (
        "We are looking for a candidate who must have kubernetes and docker. "
        "Nice to have: python scripting."
    )
    skills = ["kubernetes", "docker", "python"]

    store = _k8s_store()
    normalizer = SkillNormalizer(store, FakeEmbedderWithSkill(), FailingLLMClient())

    failing_global_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingLLMClient._Completions())
    )
    with patch.object(pipeline, "client", failing_global_client):
        # BOTH tiers fail: the injected llm_client (primary ESCO-aware classify)
        # AND classify_requirement_type (first fallback tier). The second tier
        # must reuse pipeline's real split_jd_sections.
        reqs = normalizer.normalize(jd_text, skills)

    # No exception raised; all results from the keyword fallback.
    assert len(reqs) == 3
    assert all(r.source == "keyword_fallback" for r in reqs)

    # Reuse proof: independently compute what split_jd_sections' REAL behavior
    # yields (mirroring parse_jd_requirements' own fallback branch), then assert
    # the tiers match — not a fabricated expectation.
    sections = split_jd_sections(jd_text)
    req_lower = sections["required_text"].lower()
    pref_lower = sections["preferred_text"].lower()
    exp_required = [s for s in skills if s.lower() in req_lower] or skills[:10]
    exp_preferred = [
        s for s in skills
        if s.lower() in pref_lower and s.lower() not in {r.lower() for r in exp_required}
    ]
    exp_required_set = {s.lower() for s in exp_required}

    tiered = {r.skill_name.lower(): r.tier for r in reqs}
    for s in skills:
        low = s.lower()
        if low in exp_required_set:
            assert tiered[low] == "required", "expected %r required" % s
        else:
            # Not a required cue -> either preferred by cue or the default.
            assert tiered[low] == "preferred", "expected %r preferred" % s


def test_alt_labels_for_match_and_no_match():
    store = _k8s_store()
    normalizer = SkillNormalizer(store, FakeEmbedderWithSkill(), FailingLLMClient())

    # Known concept -> its alt labels.
    assert normalizer.alt_labels_for("kubernetes") == ["container orchestration", "k8s"]
    # No match / below floor -> empty list, not an error.
    assert normalizer.alt_labels_for("quantum-xyz") == []


def test_alt_labels_for_below_floor_returns_empty():
    # A hit whose cosine is below the floor is treated as no match.
    store = make_store({"weird-term": (0.05, "DSA", ["algorithm"])})
    normalizer = SkillNormalizer(store, FakeEmbedderWithSkill(), FailingLLMClient())
    assert normalizer.alt_labels_for("weird-term") == []
    assert ONTOLOGY_MATCH_FLOOR == 0.3


if __name__ == "__main__":
    test_normalize_success_llm_classifier()
    test_normalize_llm_failure_falls_back_keyword()
    test_both_tiers_fail_reuses_split_jd_sections()
    test_alt_labels_for_match_and_no_match()
    test_alt_labels_for_below_floor_returns_empty()
    print("ALL ONTOLOGY TESTS PASSED")
