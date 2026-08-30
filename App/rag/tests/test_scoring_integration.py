"""Phase 6: RAG scoring integration tests.

Covers the wiring of Phases 1-5 into process_resumes behind
RagSettings().rag_enabled (default False). The golden-run regression asserts
that, with RAG disabled, process_resumes produces byte-identical scores to the
pre-Phase-6 baseline captured in golden_baseline.json. The remaining tests drive
the RAG path with deterministic fakes (no live LLM / Qdrant / real embedder) so
they run fast and repeatably.
"""

import hashlib
import json
import math
import os
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pipeline
from rag.config import RagSettings
from rag.retrieval import CrossEncoderReranker
from rag.store import InMemoryStore

COLL = RagSettings().resume_collection_name
ESCO_COLL = RagSettings().esco_collection_name
HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden_baseline.json")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESUMES_DIR = os.path.join(REPO, "Resumes")
JD_DIR = os.path.join(REPO, "JD")

# ---------------------------------------------------------------------------
# Lightweight deterministic fakes (no real models / LLM / Qdrant).
# ---------------------------------------------------------------------------


def _token_index(token, dim):
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % dim


def _bow(text, dim=768):
    v = [0.0] * dim
    for tok in re.findall(r"\w+", (text or "").lower()):
        v[_token_index(tok, dim)] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


class FakeEmbedder:
    """Bag-of-words hash embedder: identical/overlapping text -> high cosine."""

    def encode(self, texts):
        return [_bow(t) for t in texts]


class FakeCrossEncoder:
    """Stateful stand-in: consumes logits sequentially across predict() calls.

    The retrieval path may issue several per-request CE predictions (one per
    requirement); a sequential model mimics a real cross-encoder whose score is a
    function only of the pair, so different requirements can be made to score
    differently. If a call requests more pairs than remain, the remainder falls
    back to ``default`` (used for unpredictable span counts, never below floor).
    """

    def __init__(self, logits, default=1.5):
        self._logits = list(logits)
        self._default = default
        self._ptr = 0

    def predict(self, pairs):
        n = len(pairs)
        out = []
        for _ in range(n):
            if self._ptr < len(self._logits):
                out.append(self._logits[self._ptr])
                self._ptr += 1
            else:
                out.append(self._default)
        return out


class FakeOntologyStore:
    """Minimal VectorStore over ESCO concepts keyed by bow-similarity."""

    def __init__(self, concepts):
        # concepts: {skill: {"preferred": str, "alt": [str, ...], "score": float}}
        self._concepts = concepts

    def search_dense(self, collection, query_vector, k, filters=None):
        best = None
        best_score = -1.0
        for skill, info in self._concepts.items():
            s = _cos(query_vector, _bow(skill))
            if s > best_score:
                best_score, best = s, skill
        if best is None or best_score < self._concepts[best]["score"]:
            return []
        info = self._concepts[best]
        return [{
            "id": "concept-%s" % best,
            "score": best_score,
            "payload": {
                "preferred_label": info["preferred"],
                "alt_labels": list(info["alt"]),
                "concept_uri": "uri",
            },
        }]


class FakeScriptedLLM:
    """openai-like client that returns fixed classification / evidence verdicts.

    Dispatches on the prompt content: classification prompts contain 'Classify
    each skill below'; everything else is treated as the batched evidence prompt.
    """

    def __init__(self, classification):
        self._classification = classification
        self.calls = []
        self.chat = SimpleNamespace(completions=self._Completions(self))

    class _Completions:
        def __init__(self, owner):
            self._owner = owner

        def create(self, **kwargs):
            self._owner.calls.append(kwargs)
            content = kwargs["messages"][0]["content"]
            if "Classify each skill" in content:
                payload = json.dumps(self._owner._classification)
            else:
                payload = json.dumps(self._owner._evidence_verdicts(content))
            msg = SimpleNamespace(content=payload)
            choice = SimpleNamespace(message=msg)
            return SimpleNamespace(choices=[choice])

    def _evidence_verdicts(self, content):
        verdicts = []
        for line in content.splitlines():
            line = line.strip()
            if "SKILL:" in line:
                skill = line.split("SKILL:", 1)[1].strip()
                low = skill.lower()
                if "kuber" in low:
                    verdicts.append({
                        "skill_name": skill, "verdict": "demonstrated",
                        "cited_chunk_ids": ["exp1"], "quote": "kubernetes",
                    })
                elif "docker" in low:
                    verdicts.append({
                        "skill_name": skill, "verdict": "claimed_only",
                        "cited_chunk_ids": ["sk1"], "quote": "docker",
                    })
                else:
                    verdicts.append({
                        "skill_name": skill, "verdict": "claimed_only",
                        "cited_chunk_ids": [], "quote": None,
                    })
        return verdicts


def _index_chunks(store, candidate_id, chunks):
    points = []
    for cid, section, text in chunks:
        points.append({
            "id": cid,
            "vector": _bow(text),
            "payload": {
                "candidate_id": candidate_id,
                "chunk_id": cid,
                "section_type": section,
                "chunker_version": "v1",
                "embed_model": "test-fake",
                "raw_text": text,
            },
        })
    store.upsert(COLL, points)


def _ontology():
    return FakeOntologyStore({
        "kubernetes": {"preferred": "Kubernetes", "alt": ["k8s", "orchestration"], "score": 0.3},
        "docker": {"preferred": "Docker", "alt": ["containerization"], "score": 0.3},
    })


def _rag_runtime(ontology_store=None, resume_store=None, emb=None, logits=None,
                 classification=None, llm=None):
    store = resume_store if resume_store is not None else InMemoryStore()
    embedder = emb if emb is not None else FakeEmbedder()
    if llm is None:
        llm = FakeScriptedLLM(
            classification if classification is not None else
            {"required": ["Kubernetes", "Docker"], "preferred": []}
        )
    return {
        "ontology_store": ontology_store if ontology_store is not None else _ontology(),
        "resume_store": store,
        "embedder": embedder,
        "reranker": CrossEncoderReranker(model=FakeCrossEncoder(logits or [1.5, 1.5, 1.5])),
        "llm_client": llm,
    }


def _process_kwargs(resume_paths):
    jd_path = os.path.join(JD_DIR, "JD.pdf")
    return jd_path, resume_paths


# ---------------------------------------------------------------------------
# 1. Golden-run regression: rag_enabled=False is byte-identical to the baseline.
# ---------------------------------------------------------------------------

def test_rag_disabled_byte_identical_to_golden_baseline():
    baseline = json.load(open(GOLDEN))
    resume_files = baseline["resumes"]
    resume_paths = [os.path.join(RESUMES_DIR, f) for f in resume_files]

    with patch.dict(os.environ, {"RAG_ENABLED": "false"}):
        results = pipeline.process_resumes(
            os.path.join(JD_DIR, "JD.pdf"), resume_paths
        )

    golden = baseline["results"]
    assert len(results) == len(golden)
    for r, g in zip(results, golden):
        # Legacy score fields AND the new additive scoring_version must match.
        for key in (
            "name", "final_score", "required_skill_score", "semantic_score",
            "reranker_score", "education_score", "project_score",
            "experience_years", "matched_skills", "missing_skills",
            "scoring_version",
        ):
            assert r[key] == g[key], "field %r mismatch: got %r want %r" % (key, r[key], g[key])
        assert r["scoring_version"] == "v1-legacy"


# ---------------------------------------------------------------------------
# 2. RAG recomputes the four signals and leaves the other three untouched.
# ---------------------------------------------------------------------------

def test_rag_recomputes_signals_and_leaves_others_untouched():
    ostore = _ontology()
    rstore = InMemoryStore()
    _index_chunks(rstore, "candX", [
        ("exp1", "experience", "Managed Kubernetes clusters in production with autoscaling"),
        ("sk1", "skills", "docker, ci, containerization"),
    ])
    emb = FakeEmbedder()
    llm = FakeScriptedLLM({"required": ["Kubernetes", "Docker"], "preferred": []})

    candidate_id = "candX"
    jd_struct = {"skills": ["kubernetes", "docker"], "experience_years": 3, "education": []}
    jd_text = "must have kubernetes and docker"

    with patch.object(pipeline, "_ensure_indexed", return_value=candidate_id):
        out = pipeline.score_candidate_rag(
            jd_text, "", {"skills": []}, jd_struct, "unused.pdf",
            ostore, rstore, emb,
            CrossEncoderReranker(model=FakeCrossEncoder([1.5, 1.5])),
            llm,
        )

    # The four RAG signals are (re)computed:
    #   kubernetes demonstrated (1.0) + docker claimed_only (0.35), both required.
    assert abs(out["evidence"] - (1.0 + 0.35) / 2.0) < 1e-9
    assert out["required_skills"] == 1.0  # both required, verdict != absent
    assert out["semantic"] > 0.0  # grounded via retrieved span fused_rank
    assert out["reranker"] > 0.0  # grounded via cross-encoder score
    # matched reflects grounded skills only.
    assert set(out["matched"]) == {"kubernetes", "docker"}
    assert out["missing"] == []

    # The other three signals are NOT produced here (they stay in process_resumes
    # and are therefore unchanged by the RAG path).
    for key in ("experience", "education", "projects", "experience_years"):
        assert key not in out, "RAG signal dict must not touch %r" % key

    # Process-level check: rag on vs rag off leaves exp/edu/proj identical.
    resume = os.path.join(HERE, "fixtures", "fixture_numbered_dashes.pdf")
    jd_path = os.path.join(JD_DIR, "JD.pdf")
    with patch.object(pipeline, "_ensure_indexed", return_value=candidate_id), \
         patch.object(pipeline, "get_rag_stores", return_value={
             "ontology_store": ostore, "resume_store": rstore, "embedder": emb,
             "reranker": CrossEncoderReranker(model=FakeCrossEncoder([1.5, 1.5])),
             "llm_client": llm}), \
         patch.dict(os.environ, {"RAG_ENABLED": "true"}):
        rag_on = pipeline.process_resumes(jd_path, [resume])[0]
    with patch.dict(os.environ, {"RAG_ENABLED": "false"}):
        rag_off = pipeline.process_resumes(jd_path, [resume])[0]

    for key in ("education_score", "project_score", "experience_years", "name"):
        assert rag_on[key] == rag_off[key], "RAG must not alter %r" % key
    assert rag_on["scoring_version"] == "v1-rag"
    assert rag_off["scoring_version"] == "v1-legacy"


# ---------------------------------------------------------------------------
# 3. Preferred-tier weighting ratio (1/3).
# ---------------------------------------------------------------------------

def test_preferred_tier_weighting_ratio():
    ostore = _ontology()
    rstore = InMemoryStore()
    # Only kubernetes has genuine evidence; docker's only retrievable chunk is an
    # irrelevant spurious one that the cross-encoder rejects (logit -5 < floor),
    # so docker is genuinely absent (never sent to the LLM).
    _index_chunks(rstore, "candP", [
        ("exp1", "experience", "Managed Kubernetes clusters in production"),
    ])
    emb = FakeEmbedder()
    llm = FakeScriptedLLM({"required": ["Kubernetes"], "preferred": ["Docker"]})

    jd_struct = {"skills": ["kubernetes", "docker"], "experience_years": 3, "education": []}
    with patch.object(pipeline, "_ensure_indexed", return_value="candP"):
        out = pipeline.score_candidate_rag(
            "must have kubernetes", "", {"skills": []}, jd_struct,
            "unused.pdf", ostore, rstore, emb,
            CrossEncoderReranker(model=FakeCrossEncoder([1.5, -5.0])),
            llm,
        )

    # required kubernetes demonstrated (tw=1), preferred docker absent (tw=1/3):
    #   evidence = (1*1.0 + 1/3*0.0) / (1 + 1/3) = 1 / (4/3) = 0.75
    ratio = pipeline.PREFERRED_TIER_WEIGHT_RATIO
    assert abs(ratio - (1.0 / 3.0)) < 1e-9
    assert abs(out["evidence"] - (1.0 + 0.0 * ratio) / (1.0 + ratio)) < 1e-9
    # required_skills gate: only the demonstrated required skill counts.
    assert abs(out["required_skills"] - 1.0 / (1.0 + ratio)) < 1e-9
    assert out["missing"] == ["docker"]
    assert out["matched"] == ["kubernetes"]


# ---------------------------------------------------------------------------
# 4. Any RAG-path exception falls back to legacy for that candidate only.
# ---------------------------------------------------------------------------

def test_rag_exception_falls_back_to_legacy():
    class ExplodingOntology:
        def search_dense(self, *a, **k):
            raise RuntimeError("ontology store blew up")

    resume = os.path.join(HERE, "fixtures", "fixture_numbered_dashes.pdf")
    jd_path = os.path.join(JD_DIR, "JD.pdf")

    runtime = _rag_runtime()
    runtime["ontology_store"] = ExplodingOntology()

    with patch.object(pipeline, "get_rag_stores", return_value=runtime), \
         patch.dict(os.environ, {"RAG_ENABLED": "true"}):
        rag_on = pipeline.process_resumes(jd_path, [resume])[0]
    with patch.dict(os.environ, {"RAG_ENABLED": "false"}):
        rag_off = pipeline.process_resumes(jd_path, [resume])[0]

    # Candidate fell back to the exact legacy path, so it matches byte-for-byte.
    assert rag_on["scoring_version"] == "v1-legacy"
    assert rag_on["final_score"] == rag_off["final_score"]
    assert rag_on["required_skill_score"] == rag_off["required_skill_score"]


# ---------------------------------------------------------------------------
# 5. scoring_version correctness.
# ---------------------------------------------------------------------------

def test_scoring_version_markers():
    resume = os.path.join(HERE, "fixtures", "fixture_numbered_dashes.pdf")
    jd_path = os.path.join(JD_DIR, "JD.pdf")
    candidate_id = "candV"
    rstore = InMemoryStore()
    _index_chunks(rstore, candidate_id, [("exp1", "experience", "kubernetes platforms")])
    emb = FakeEmbedder()
    llm = FakeScriptedLLM({"required": ["Kubernetes"], "preferred": []})
    ostore = _ontology()

    with patch.object(pipeline, "_ensure_indexed", return_value=candidate_id), \
         patch.object(pipeline, "get_rag_stores", return_value={
             "ontology_store": ostore, "resume_store": rstore, "embedder": emb,
             "reranker": CrossEncoderReranker(model=FakeCrossEncoder([1.5])),
             "llm_client": llm}), \
         patch.dict(os.environ, {"RAG_ENABLED": "true"}):
        rag_on = pipeline.process_resumes(jd_path, [resume])[0]

    with patch.dict(os.environ, {"RAG_ENABLED": "false"}):
        rag_off = pipeline.process_resumes(jd_path, [resume])[0]

    assert rag_on["scoring_version"] == "v1-rag"
    assert rag_off["scoring_version"] == "v1-legacy"


# ---------------------------------------------------------------------------
# 6. Indexing idempotence: a repeated ingest does not re-embed/index.
# ---------------------------------------------------------------------------

def test_indexing_idempotent_for_repeated_request():
    pipeline._rag_indexed_candidate_ids.clear()

    store = InMemoryStore()
    emb = FakeEmbedder()
    resume = os.path.join(HERE, "fixtures", "fixture_numbered_dashes.pdf")

    first = pipeline._ensure_indexed(resume, store, emb)
    count_after_first = store.count(COLL)
    assert count_after_first > 0

    second = pipeline._ensure_indexed(resume, store, emb)
    count_after_second = store.count(COLL)

    # Deterministic candidate id and NO re-index on the second request.
    assert first == second
    assert count_after_second == count_after_first


if __name__ == "__main__":
    test_rag_disabled_byte_identical_to_golden_baseline()
    test_rag_recomputes_signals_and_leaves_others_untouched()
    test_preferred_tier_weighting_ratio()
    test_rag_exception_falls_back_to_legacy()
    test_scoring_version_markers()
    test_indexing_idempotent_for_repeated_request()
    print("ALL SCORING INTEGRATION TESTS PASSED")
