"""Phase 4: hybrid retrieval, RRF fusion, cross-encoder reranking.

Builds on Phase 3's ``VectorStore`` and ``EmbeddingService``.

CROSS-ENCODER LOADING DECISION
------------------------------
``CrossEncoderReranker`` instantiates its OWN :class:`sentence_transformers.CrossEncoder`
rather than importing the ``reranker_model`` already loaded at ``app/pipeline.py``'s
module scope. Importing from ``pipeline.py`` would trigger that module's import-time
side effects (FastAPI client construction, spaCy model loading) which this retrieval
package should not depend on. Keeping a self-contained instance also makes these
tests independent of ``pipeline.py``'s import behavior. An optional ``model`` can be
injected for testability (a lightweight stand-in) so the suite does not have to load
the real cross-encoder.

The model identity is unchanged from the pipeline: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
(the spec forbids swapping it).
"""

from __future__ import annotations

import math

from sentence_transformers import CrossEncoder

from rag.config import RagSettings
from rag.embedding import EmbeddingService
from rag.schemas import EvidenceSpan, NormalizedRequirement
from rag.store import VectorStore

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Skills section down-weight: a keyword dump is a claim, not evidence, so a
# skills chunk's fused score is halved before final ranking.
SKILLS_SECTION_WEIGHT = 0.5

# Cross-encoder relevance floor applied by SkillEvidenceRetriever. The squash
# 1/(1+exp(-s/1.5)) equals 0.5 exactly when the raw logit s == 0, i.e. the
# model is neutral. Requiring > 0.5 means we only count pairs the model leans
# positive on; an empty result is a legitimate "absent", never forced.
CE_RELEVANCE_FLOOR = 0.5


class RRFFuser:
    """Reciprocal Rank Fusion over multiple ranked ID lists.

    score(doc) = sum(1 / (k + rank_in_list_i)) across all lists that contain
    the doc. A document present in many lists (or high in one list) scores
    higher than one appearing in a single list.
    """

    def fuse(self, rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    """Runs dense and sparse retrieval per query, fuses via RRF, returns top-k
    evidence spans with the skills section down-weighted."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingService,
        fuser: RRFFuser,
    ):
        self._store = store
        self._embedder = embedder
        self._fuser = fuser
        self._settings = RagSettings()

    def _payload_text(self, hit: dict) -> str:
        return (
            hit["payload"].get("raw_text")
            or hit["payload"].get("text")
            or hit["payload"].get("preferred_label")
            or ""
        )

    def _chunk(self, hit: dict) -> EvidenceSpan:
        payload = hit["payload"]
        return EvidenceSpan(
            chunk_id=payload.get("chunk_id") or hit["id"],
            candidate_id=payload.get("candidate_id", ""),
            text=self._payload_text(hit),
        )

    def _rank_fields(self, spans: list[EvidenceSpan], hitmap: dict) -> list[EvidenceSpan]:
        """Best-effort fill of dense_rank/lexical_rank from the raw hit maps."""
        for span in spans:
            info = hitmap.get(span.chunk_id)
            if info:
                span.dense_rank = info.get("dense")
                span.lexical_rank = info.get("sparse")
        return spans

    def retrieve(
        self, queries: list[str], candidate_id: str, k: int = 8
    ) -> list[EvidenceSpan]:
        # Collect every ranking from every query's dense and sparse searches,
        # then fuse them ALL together in one RRF pass. For a single query this
        # degenerates to fusing dense+sparse (exactly the per-query case); for
        # multiple queries (Idea B alt-labels) it aggregates all queries'
        # rankings simultaneously. Query strings are never concatenated.
        dense_filter = {"candidate_id": candidate_id}
        rankings: list[list[str]] = []
        hitmap: dict[str, dict] = {}
        raw_hits: dict[str, dict] = {}

        query_vectors = self._embedder.encode(queries)

        for i, query in enumerate(queries):
            qvec = query_vectors[i]

            dense_hits = self._store.search_dense(
                self._settings.resume_collection_name,
                qvec,
                k=self._settings.dense_top_k,
                filters=dense_filter,
            )
            sparse_hits = self._store.search_sparse(
                self._settings.resume_collection_name,
                query,
                k=self._settings.lexical_top_k,
                filters=dense_filter,
            )

            dense_ids = [h["id"] for h in dense_hits]
            sparse_ids = [h["id"] for h in sparse_hits]
            rankings.append(dense_ids)
            rankings.append(sparse_ids)

            for rank, h in enumerate(dense_hits):
                hitmap.setdefault(h["id"], {}).setdefault("dense", rank + 1)
                raw_hits[h["id"]] = h
            for rank, h in enumerate(sparse_hits):
                hitmap.setdefault(h["id"], {}).setdefault("sparse", rank + 1)
                raw_hits.setdefault(h["id"], h)

        fused = self._fuser.fuse(rankings, k=60)

        spans: list[EvidenceSpan] = []
        for chunk_id, score in fused:
            h = raw_hits.get(chunk_id)
            if h is None:
                continue
            span = self._chunk(h)
            span.fused_rank = score
            spans.append(span)

        # Skills-section down-weight: halve the fused score of skills chunks
        # so a keyword dump does not dominate retrieval for every requirement.
        for span in spans:
            section = raw_hits[span.chunk_id]["payload"].get("section_type")
            if section == "skills":
                span.fused_rank = span.fused_rank * SKILLS_SECTION_WEIGHT

        spans.sort(key=lambda s: -s.fused_rank)
        spans = self._rank_fields(spans, hitmap)
        return spans[:k]


class CrossEncoderReranker:
    """Reranks evidence spans against a query using a cross-encoder."""

    def __init__(self, model=None):
        self._injected = model is not None
        self._model = model
        self._settings = RagSettings()

    def _load(self):
        if self._model is None:
            # Self-contained instance (see module docstring): do NOT import
            # pipeline.py's reranker_model to avoid its import-time side effects.
            self._model = CrossEncoder(_CROSS_ENCODER_MODEL)
        return self._model

    def _squash(self, logit: float) -> float:
        # Recalibrated squash for CHUNK-level pairs. This is the same formula
        # as pipeline.py's compute_reranker_score (1/(1+exp(-s/1.5))), which
        # was tuned for full-document pairs. It is reused here as the starting
        # point (rather than inventing a new one from scratch) but its
        # temperature (1.5) SHOULD be re-validated against the Phase 9 gold
        # set once that exists, since chunk-level pairs score differently than
        # whole-document pairs.
        return 1.0 / (1.0 + math.exp(-logit / 1.5))

    def rerank(
        self, query: str, spans: list[EvidenceSpan], top_n: int = 3
    ) -> list[EvidenceSpan]:
        if not spans:
            return []
        model = self._load()
        pairs = [(query, span.text) for span in spans]
        logits = model.predict(pairs)
        for span, logit in zip(spans, logits):
            span.cross_encoder_score = self._squash(float(logit))
        order = sorted(spans, key=lambda s: s.cross_encoder_score, reverse=True)
        return order[:top_n]


class SkillEvidenceRetriever:
    """Retrieves up to 3 evidence spans supporting a single requirement."""

    def __init__(self, retriever: HybridRetriever, reranker: CrossEncoderReranker):
        self._retriever = retriever
        self._reranker = reranker

    def for_requirement(
        self,
        requirement: NormalizedRequirement,
        candidate_id: str,
        alt_labels: list[str] | None = None,
    ) -> list[EvidenceSpan]:
        # alt_labels accepted now (from the future ESCO lookup in Phase 5);
        # Phase 5 is what will actually supply real alt_labels. Build the
        # multi-query list from the skill name plus any provided labels.
        queries = [requirement.skill_name] + list(alt_labels or [])
        spans = self._retriever.retrieve(queries, candidate_id, k=8)

        # Rerank against the canonical skill name; up to 3 spans.
        reranked = self._reranker.rerank(requirement.skill_name, spans, top_n=3)

        # Relevance floor: only keep spans the model leans positive on. Do NOT
        # lower this to force a match. An empty result means "absent", which is
        # legitimate and expected. (When no model was injected — real CE path —
        # floor applies to the real cross-encoder score.)
        kept = [s for s in reranked if (s.cross_encoder_score or 0.0) >= CE_RELEVANCE_FLOOR]
        return kept
