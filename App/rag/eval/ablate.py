"""Retrieval ablation harness.

Implements a config-driven ablation over the retrieval pipeline so retrieval
parameters can be tuned in Phase 4 against whatever gold evidence exists. Note
that a REAL gold set only arrives in Phase 9; the harness is nonetheless built
here (per spec) and exercised in tests with a small, clearly-labeled SYNTHETIC
gold set purely to prove the harness works end-to-end.
"""

from __future__ import annotations

from rag.config import RagSettings
from rag.embedding import EmbeddingService
from rag.retrieval import CrossEncoderReranker, HybridRetriever, RRFFuser
from rag.store import VectorStore


class RetrievalAblation:
    """Runs recall@3 for a set of retrieval configs against a gold set.

    gold_pairs: list of dicts like
        {"requirement": str, "candidate_id": str, "gold_chunk_id": str}
    configs: subset of {"dense_only", "bm25_only", "hybrid", "hybrid_ce"}.

    Returns {config_name: recall@3} where recall@3 is the fraction of gold
    pairs whose gold_chunk_id appears in the top-3 spans returned for that
    requirement/candidate.
    """

    def __init__(self, store, embedder, reranker=None):
        self._store = store
        self._embedder = embedder
        self._reranker = reranker or CrossEncoderReranker()
        self._coll = RagSettings().resume_collection_name

    def _top3_ids(self, mode: str, query: str, candidate_id: str) -> list:
        dense_filter = {"candidate_id": candidate_id}

        if mode == "dense_only":
            qvec = self._embedder.encode([query])[0]
            hits = self._store.search_dense(
                self._coll, qvec, k=8, filters=dense_filter
            )
            return [h["id"] for h in hits[:3]]

        if mode == "bm25_only":
            hits = self._store.search_sparse(
                self._coll, query, k=8, filters=dense_filter
            )
            return [h["id"] for h in hits[:3]]

        retriever = HybridRetriever(self._store, self._embedder, RRFFuser())
        spans = retriever.retrieve([query], candidate_id, k=8)
        if mode == "hybrid_ce":
            spans = self._reranker.rerank(query, spans, top_n=3)
        return [s.chunk_id for s in spans[:3]]

    def run(self, gold_pairs: list[dict], configs: list[str] | None = None) -> dict:
        if configs is None:
            configs = ["dense_only", "bm25_only", "hybrid", "hybrid_ce"]
        results = {}
        for cfg in configs:
            hit = 0
            for pair in gold_pairs:
                top3 = self._top3_ids(cfg, pair["requirement"], pair["candidate_id"])
                if pair["gold_chunk_id"] in top3:
                    hit += 1
            results[cfg] = hit / len(gold_pairs) if gold_pairs else 0.0
        return results
