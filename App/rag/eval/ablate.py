"""Retrieval ablation harness.

Implements a config-driven ablation over the retrieval pipeline so retrieval
parameters can be tuned in Phase 4 against whatever gold evidence exists. A REAL
gold set arrives in Phase 9; the harness is nonetheless used in tests with a
small, clearly-labeled SYNTHETIC gold set purely to prove the harness works
end-to-end.

Phase 9 extension: six rows are requested —
  dense_only, bm25_only, hybrid, hybrid_ce,
  hybrid_ce+header_prefix, hybrid_ce+ontology_expansion.
``hybrid_ce+ontology_expansion`` is real (it runs the same multi-query alt-label
retrieval ``SkillEvidenceRetriever.for_requirement`` uses). ``hybrid_ce+header_prefix``
CANNOT run: Phase 2's ``prefix_context`` is explicitly NOT wired into chunking or
indexing (chunking.py keeps it "reserved for the embedding/retrieval string-
construction step in a later phase"), so the embedding text of every indexed
chunk is header-free today. Running that row would fabricate numbers; we surface
it as a documented gap instead.
"""

from __future__ import annotations

from rag.config import RagSettings
from rag.embedding import EmbeddingService
from rag.retrieval import CrossEncoderReranker, HybridRetriever, RRFFuser
from rag.store import VectorStore


class RetrievalAblation:
    """Runs recall@3 for a set of retrieval configs against a gold set.

    gold_pairs: list of dicts like
        {"requirement": str, "candidate_id": str, "gold_chunk_id": str,
         "requirement_id": str (Phase 9, optional but recommended)}
    configs: subset of
        {"dense_only", "bm25_only", "hybrid", "hybrid_ce",
         "hybrid_ce+ontology_expansion", "hybrid_ce+header_prefix"}.

    Returns {config_name: {"recall@3": float, "pairs": int, "hit": int}}. For
    rows that cannot be measured honestly the value is
    {"recall@3": None, "gap": "...reason..."}.
    """

    # Header prefixing is not wired into the embedding-text construction step
    # (chunking.prefix_context is reserved, never called by chunk()/ingest());
    # this config therefore has no implementation to measure.
    CONFIG_HEADER_PREFIX_GAP = (
        "hybrid_ce+header_prefix: prefix_context() is not wired into chunking/"
        "indexing (chunking.py reserves it for a later phase), so no chunk is "
        "embedded with a section header today; this row cannot be measured "
        "without changing the ingest path (out of scope for evaluation phase)."
    )

    def __init__(self, store, embedder, reranker=None, alt_labels_provider=None):
        self._store = store
        self._embedder = embedder
        self._reranker = reranker or CrossEncoderReranker()
        # Callable requirement_text -> list[str] fed by an ESCO-backed normalizer.
        self._alt_labels_provider = alt_labels_provider
        self._coll = RagSettings().resume_collection_name

    def _alt_labels_for(self, query: str) -> list[str]:
        if self._alt_labels_provider is None:
            return []
        try:
            return self._alt_labels_provider(query) or []
        except Exception:
            return []

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

        if mode in ("hybrid", "hybrid_ce"):
            queries = [query]
            spans = retriever.retrieve(queries, candidate_id, k=8)
        elif mode == "hybrid_ce+ontology_expansion":
            queries = [query] + self._alt_labels_for(query)
            spans = retriever.retrieve(queries, candidate_id, k=8)
            spans = self._reranker.rerank(query, spans, top_n=3)
            return [s.chunk_id for s in spans[:3]]
        elif mode == "hybrid_ce+header_prefix":
            raise ValueError(self.CONFIG_HEADER_PREFIX_GAP)
        else:
            raise ValueError("unknown config %r" % mode)

        if mode == "hybrid_ce":
            spans = self._reranker.rerank(query, spans, top_n=3)
        return [s.chunk_id for s in spans[:3]]

    def run_with_topk(self, gold_pairs: list[dict], configs: list[str] | None = None) -> dict:
        """Return {config: {(requirement_id_or_text, candidate_id): top3_ids}}."""
        if configs is None:
            configs = ["dense_only", "bm25_only", "hybrid", "hybrid_ce"]
        out = {}
        for cfg in configs:
            per_pair = {}
            for pair in gold_pairs:
                key = (
                    pair.get("requirement_id", pair["requirement"]),
                    pair["candidate_id"],
                )
                try:
                    per_pair[key] = self._top3_ids(
                        cfg, pair["requirement"], pair["candidate_id"]
                    )
                except ValueError as exc:
                    per_pair[key] = {"gap": str(exc)}
            out[cfg] = per_pair
        return out

    def run(self, gold_pairs: list[dict], configs: list[str] | None = None) -> dict:
        results = {}
        for cfg, per_pair in self.run_with_topk(gold_pairs, configs).items():
            hit = total = 0
            gap = None
            for (rid, cid), top3 in per_pair.items():
                target = next(
                    (p for p in gold_pairs if p.get("requirement_id", p["requirement"]) == rid and p["candidate_id"] == cid),
                    None,
                )
                if target is None:
                    continue
                if isinstance(top3, dict) and top3.get("gap"):
                    gap = top3["gap"]
                    continue
                total += 1
                if target["gold_chunk_id"] in top3:
                    hit += 1
            if gap is not None and total == 0:
                results[cfg] = {"recall@3": None, "gap": gap}
            else:
                results[cfg] = {
                    "recall@3": (hit / total) if total else 0.0,
                    "pairs": total,
                    "hit": hit,
                }
        return results