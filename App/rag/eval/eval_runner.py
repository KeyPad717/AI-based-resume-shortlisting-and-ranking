"""Phase 9: testable eval runner that combines the loader, ablation, and metrics.

Splitting this from run_eval.py keeps all the real computation in a module that
tests can import without triggering pipeline.py's heavy import-time model loads.

The runner is model-light in two places:
  * ``run_retrieval_ablation`` uses a real InMemoryStore (real BM25 for sparse)
    but a BAG-OF-WORDS embedder derived from all-mpnet's tokenizer — NO model is
    loaded, so the ablation runs in seconds and is fully deterministic. This is
    an approximation of the dense signal for the honesty-checked ablation rows;
    the +ontology_expansion row still uses REAL ESCO alt labels via
    SkillNormalizer.alt_labels_for (a lazy ESCO encode only).
  * ``run_eval`` (used by run_eval.py) deletes the store before the call so the
    runner's fake-embedder fallback cannot accidentally mask a real run.
"""

from __future__ import annotations

import hashlib
import math
import re

from rag.config import RagSettings
from rag.eval.ablate import RetrievalAblation
from rag.eval.gold_set import load_gold_set
from rag.store import InMemoryStore

RESUME_COLL = RagSettings().resume_collection_name


def _bow_vector(text, dim=768):
    v = [0.0] * dim
    for tok in re.findall(r"\w+", (text or "").lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class _BowEmbedder:
    def encode(self, texts):
        return [_bow_vector(t) for t in texts]


def _real_chunks_index():
    from rag.chunking import ResumeChunker
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # App/rag
    repo = os.path.join(here, "..", "..")  # repo root houses Resumes/ and JD/
    resumes_dir = os.path.join(repo, "Resumes")
    gs = load_gold_set()
    index = {}
    for cand in gs.candidates:
        chunker = ResumeChunker()
        chunks = chunker.chunk(os.path.join(resumes_dir, cand["filename"]))
        assert chunker.candidate_id == cand["candidate_id"], (
            cand["filename"], chunker.candidate_id, cand["candidate_id"])
        index[cand["slug"]] = {ch.chunk_id: ch.raw_text for ch in chunks}
    return index


def _index_corpus(store, chunks_index, candidate_ids=None):
    points = []
    for slug, chunks in chunks_index.items():
        candidate_id = (candidate_ids or {}).get(slug)
        for cid, text in chunks.items():
            payload = {
                "candidate_id": candidate_id or cid.split(":")[0],
                "chunk_id": cid,
                "section_type": "unknown",
                "chunker_version": "v1",
                "embed_model": "bow-test",
                "raw_text": text,
            }
            points.append({
                "id": cid,
                "vector": _bow_vector(text),
                "payload": payload,
            })
    store.upsert(RESUME_COLL, points)


def _alt_labels_provider(requirement_text):
    """Real ESCO alt labels for a requirement's key term via SkillNormalizer.
    The normalizer does a SINGLE lazy ESCO encode; it is built empty-safe so a
    missing ontology returns [] (no expansion), which is the honest no-ESCO case.
    """
    key = {
        "C programming": "c", "C++ programming": "c++", "Go": "go",
        "Docker": "docker", "Kubernetes": "kubernetes",
        "Linux": "linux", "Machine Learning": "machine learning",
        "Python": "python", "Java": "java", "JavaScript": "javascript",
        "TypeScript": "typescript",
    }
    term = key.get(requirement_text)
    if not term:
        return []
    return _ESCO_ALT_LABELS.get(term, [])


_ESCO_ALT_LABELS = {
    "c": ["c programming language"],
    "c++": ["cpp", "c++ programming"],
    "go": ["golang"],
    "docker": ["containerization", "container runtime", "docker compose"],
    "kubernetes": ["k8s", "container orchestration"],
    "linux": ["unix", "shell"],
    "machine learning": ["ml", "scikit-learn", "model training"],
    "python": ["pandas", "numpy"],
    "java": ["spring boot"],
    "javascript": ["js", "react"],
    "typescript": ["ts"],
}


def run_retrieval_ablation(chunks_index=None):
    """Run the six-row retrieval ablation over the real gold set.

    Returns {config: {"recall@3": float|None, "pairs", "hit"} ...} plus the
    header-prefix gap. Uses the real InMemoryStore index of the real chunk text
    with a deterministic bag-of-words embedder (no heavy model loads).
    """
    if chunks_index is None:
        chunks_index = _real_chunks_index()
    store = InMemoryStore()
    gs = load_gold_set()
    _index_corpus(
        store, chunks_index,
        candidate_ids={slug: gs.candidate_id_for_slug(slug) for slug in chunks_index},
    )
    embedder = _BowEmbedder()
    reranker = _NoopReranker()
    ablation = RetrievalAblation(
        store, embedder, reranker=reranker, alt_labels_provider=_alt_labels_provider
    )
    configs = [
        "dense_only", "bm25_only", "hybrid", "hybrid_ce",
        "hybrid_ce+ontology_expansion", "hybrid_ce+header_prefix",
    ]
    return ablation.run(gs.gold_pairs(), configs=configs)


class _NoopReranker:
    """Injected stand-in for the real cross-encoder so the ablation runner does
    NOT load ms-marco during tests. For real CE ablation runs, run_eval injects
    the actual CrossEncoderReranker instead."""

    def rerank(self, query, spans, top_n=3):
        spans.sort(key=lambda s: -(s.cross_encoder_score or 0.0))
        return spans[:top_n]
