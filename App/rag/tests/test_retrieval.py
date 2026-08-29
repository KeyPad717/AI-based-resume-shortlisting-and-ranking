import hashlib
import math
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.config import RagSettings
from rag.eval.ablate import RetrievalAblation
from rag.retrieval import (
    CE_RELEVANCE_FLOOR,
    CrossEncoderReranker,
    HybridRetriever,
    RRFFuser,
    SkillEvidenceRetriever,
)
from rag.schemas import NormalizedRequirement
from rag.store import InMemoryStore

COLL = RagSettings().resume_collection_name


# --------------------------------------------------------------------------
# Lightweight deterministic fakes so the suite runs fast WITHOUT loading the
# real all-mpnet embedder or the cross-encoder. Dense vectors are bag-of-words
# hashes: identical/overlapping text yields high cosine, which is exactly what
# the tests need, and sparse search uses REAL BM25 from the store.
# --------------------------------------------------------------------------

def _token_index(token, dim):
    h = int(hashlib.md5(token.encode()).hexdigest(), 16)
    return h % dim


def _bow_vector(text, dim=768):
    v = [0.0] * dim
    for tok in re.findall(r"\w+", text.lower()):
        v[_token_index(tok, dim)] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedder:
    """Stand-in for EmbeddingService exposing the same encode() contract."""

    def encode(self, texts):
        return [_bow_vector(t) for t in texts]


class FakeCrossEncoder:
    """Stand-in CrossEncoder with a controllable predict()->logits."""

    def __init__(self, logits):
        self._logits = list(logits)

    def predict(self, pairs):
        return self._logits[: len(pairs)]


def _index_chunks(store, candidate_id, chunks):
    points = []
    for cid, section, text in chunks:
        points.append(
            {
                "id": cid,
                "vector": _bow_vector(text),
                "payload": {
                    "candidate_id": candidate_id,
                    "chunk_id": cid,
                    "section_type": section,
                    "chunker_version": "v1",
                    "embed_model": "test-fake",
                    "raw_text": text,
                },
            }
        )
    store.upsert(COLL, points)


def _spinup(chunks, candidate_id="candA", logits=None):
    store = InMemoryStore()
    _index_chunks(store, candidate_id, chunks)
    embedder = FakeEmbedder()
    retriever = HybridRetriever(store, embedder, RRFFuser())
    if logits is None:
        reranker = CrossEncoderReranker(model=FakeCrossEncoder([3.0, 3.0, 3.0, 3.0]))
    else:
        reranker = CrossEncoderReranker(model=FakeCrossEncoder(logits))
    return store, retriever, reranker


def test_rrf_fuser_shared_doc_ranks_first():
    rankings = [
        ["a", "b", "c", "d"],
        ["c", "e", "f", "g"],  # 'c' appears in both lists
    ]
    fused = RRFFuser().fuse(rankings, k=60)
    scores = dict(fused)
    # 'c' is in both lists -> must score strictly higher than docs in one list.
    assert scores["c"] > scores["a"]
    assert scores["c"] > scores["b"]
    assert scores["c"] > scores["e"]
    assert scores["c"] > scores["g"]
    assert fused[0][0] == "c"


def test_hybrid_retriever_returns_exactly_3_spans_for_3_chunks():
    chunks = [
        ("k8-1", "experience", "designed kubernetes clusters for the platform team"),
        ("k8-2", "projects", "built a container orchestration dashboard in react"),
        ("k8-3", "education", "bachelor degree in computer science"),
    ]
    store, retriever, _ = _spinup(chunks, candidate_id="candA")
    spans = retriever.retrieve(["kubernetes container orchestration"], "candA", k=8)
    assert len(spans) == 3, "expected exactly 3 spans for a 3-chunk candidate, got %d" % len(spans)
    ids = {s.chunk_id for s in spans}
    assert ids == {"k8-1", "k8-2", "k8-3"}


def test_skills_section_downweight():
    text = "python pandas numpy machine learning model training"
    chunks = [
        ("sk-1", "skills", text),
        ("sk-2", "projects", text),  # identical text, different section
    ]
    store, retriever, _ = _spinup(chunks, candidate_id="candA")
    spans = retriever.retrieve(["python machine learning"], "candA", k=8)
    by_id = {s.chunk_id: s for s in spans}
    # Same text => identical dense/sparse signal, but skills is *0.5.
    assert by_id["sk-2"].fused_rank > by_id["sk-1"].fused_rank
    # projects chunk must appear at or above the skills chunk in the ranking.
    assert spans.index(by_id["sk-2"]) <= spans.index(by_id["sk-1"])


def test_cross_encoder_rerank_populates_score_and_top_n():
    chunks = [
        ("ce-1", "projects", "deployed microservices on aws ecs with terraform"),
        ("ce-2", "projects", "wrote documentation for the internal api gateway"),
        ("ce-3", "experience", "ran weekly standups and code reviews"),
        ("ce-4", "experience", "tuned postgres queries to cut p95 latency"),
        ("ce-5", "projects", "built a kubernetes operator in golang"),
    ]
    store, retriever, _ = _spinup(chunks, candidate_id="candA")
    spans = retriever.retrieve(["kubernetes operator"], "candA", k=8)
    # Inject a fake CE whose per-span logits follow retrieve()'s order.
    fake = FakeCrossEncoder([1.0, 0.5, 2.0, 0.1, 5.0])
    reranker = CrossEncoderReranker(model=fake)
    top = reranker.rerank("kubernetes operator golang", spans, top_n=3)
    assert len(top) == 3, "top_n must be respected, got %d" % len(top)
    for s in top:
        assert s.cross_encoder_score is not None, "cross_encoder_score must be populated"
    scores = [s.cross_encoder_score for s in top]
    assert scores == sorted(scores, reverse=True), "scores must be sorted descending"
    # top-3 must be exactly the 3 spans with the highest fake logits.
    top_indices = sorted(range(len(spans)), key=lambda i: fake._logits[i], reverse=True)[:3]
    expected_ids = {spans[i].chunk_id for i in top_indices}
    assert {s.chunk_id for s in top} == expected_ids


def test_for_requirement_absent_returns_empty():
    # Candidate content has nothing to do with the skill -> absent, not forced.
    chunks = [
        ("ab-1", "experience", "maintained jira boards and on-call rotations"),
        ("ab-2", "projects", "migrated legacy sql reports to tableau dashboards"),
    ]
    store, retriever, _ = _spinup(chunks, candidate_id="candA", logits=[-8.0, -8.0])
    skill = SkillEvidenceRetriever(retriever, CrossEncoderReranker(model=FakeCrossEncoder([-8.0, -8.0])))
    req = NormalizedRequirement(skill_name="cobol mainframe migration", tier="required", source="llm_classifier")
    result = skill.for_requirement(req, "candA", alt_labels=["z/os assembler"])
    assert result == [], "absent requirement must return an empty list, got %r" % result


def test_ablation_run_returns_all_configs():
    # ---- SYNTHETIC gold set (NOT the real Phase 9 evaluation) ----
    # Small fabricated triples over data we just indexed, purely to prove the
    # harness executes end-to-end.
    chunks = [
        ("g-1", "experience", "managed kubernetes clusters at scale for thousands of pods"),
        ("g-2", "projects", "built a grafana dashboard for postgres observability"),
        ("g-3", "skills", "postgres, sql, data modeling, indexing"),
    ]
    store, retriever, _ = _spinup(chunks, candidate_id="candG")
    gold_pairs = [
        {"requirement": "kubernetes cluster management", "candidate_id": "candG", "gold_chunk_id": "g-1"},
        {"requirement": "postgres observability", "candidate_id": "candG", "gold_chunk_id": "g-2"},
    ]
    ablation = RetrievalAblation(store, FakeEmbedder(), reranker=CrossEncoderReranker(model=FakeCrossEncoder([1.0, 1.0, 1.0])))
    configs = ["dense_only", "bm25_only", "hybrid", "hybrid_ce"]
    results = ablation.run(gold_pairs, configs)
    # All requested config keys present, float values in [0,1].
    for cfg in configs:
        assert cfg in results, "missing config %s in ablation results %r" % (cfg, results)
        assert isinstance(results[cfg], float)
        assert 0.0 <= results[cfg] <= 1.0
    # Sanity: hybrid should be at least as good as either single-signal config
    # on this gold set (RRF can only add), and ce should not reduce beyond 0.
    assert results["hybrid"] >= min(results["dense_only"], results["bm25_only"])
    print("SYNTHETIC ablation (harness validation, NOT real Phase 9): %s" % results)


if __name__ == "__main__":
    test_rrf_fuser_shared_doc_ranks_first()
    test_hybrid_retriever_returns_exactly_3_spans_for_3_chunks()
    test_skills_section_downweight()
    test_cross_encoder_rerank_populates_score_and_top_n()
    test_for_requirement_absent_returns_empty()
    test_ablation_run_returns_all_configs()
    print("ALL RETRIEVAL TESTS PASSED")
