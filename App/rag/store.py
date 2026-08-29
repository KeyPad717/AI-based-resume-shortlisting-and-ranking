import logging
import re
from abc import ABC, abstractmethod

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except Exception:  # pragma: no cover - rank_bm25 optional for pure fallback
    _BM25_AVAILABLE = False

try:
    import qdrant_client
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        FilterSelector,
        MatchAny,
        PointStruct,
        VectorParams,
    )
    _QDRANT_AVAILABLE = True
except Exception:  # pragma: no cover - qdrant-client optional for in-memory use
    _QDRANT_AVAILABLE = False


logger = logging.getLogger(__name__)


def _tokenize(text):
    return set(re.findall(r"\w+", (text or "").lower()))


def _bm25_corpus_tokens(corpus_texts):
    """Tokenize a list of texts for BM25Okapi.

    Each text becomes a list of lowercased word tokens. BM25Okapi expects a
    list of such lists. Empty docs become empty token lists (rank_bm25 skips
    them during scoring, so they stay placed but never match).
    """
    return [re.findall(r"\w+", (t or "").lower()) for t in corpus_texts]


def _bm25_scores(bm25, query_text):
    """Return {doc_index: score} from a BM25Okapi index. Raises a clear error
    if rank_bm25 could not be imported."""
    if not _BM25_AVAILABLE:
        raise RuntimeError(
            "rank_bm25 is required for BM25 sparse retrieval; "
            "install it via 'pip install rank-bm25'."
        )
    qtokens = re.findall(r"\w+", (query_text or "").lower())
    row = bm25.get_scores(qtokens)
    return {i: float(s) for i, s in enumerate(row)}


def _matches(payload, filters):
    if not filters:
        return True
    for key, value in filters.items():
        if key not in payload:
            return False
        if isinstance(value, (list, tuple)):
            if payload[key] not in value:
                return False
        elif payload[key] != value:
            return False
    return True


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, collection, points):
        ...

    @abstractmethod
    def search_dense(self, collection, query_vector, k, filters=None):
        ...

    @abstractmethod
    def search_sparse(self, collection, query_text, k, filters=None):
        ...

    @abstractmethod
    def delete_by(self, collection, filters):
        ...

    @abstractmethod
    def count(self, collection):
        ...

    @abstractmethod
    def health(self):
        ...

    @staticmethod
    def _result_list(items):
        return [
            {"id": pid, "score": score, "payload": payload}
            for pid, score, payload in items
        ]


class InMemoryStore(VectorStore):
    """Pure Python/numpy vector store. Dict-of-lists per collection.

    Point shape (shared with QdrantStore so the same points dict works for
    both stores): {"id": <str>, "vector": [float, ...], "payload": {...}}.
    """

    def __init__(self, expected_dimension=None):
        self._data = {}
        self._expected_dimension = expected_dimension

    def _collection(self, collection):
        return self._data.setdefault(collection, {})

    def upsert(self, collection, points):
        coll = self._collection(collection)
        for point in points:
            vector = point["vector"]
            if self._expected_dimension is not None and len(vector) != self._expected_dimension:
                raise ValueError(
                    "vector dimension %d does not match expected dimension %d"
                    % (len(vector), self._expected_dimension)
                )
            coll[point["id"]] = {
                "id": point["id"],
                "vector": vector,
                "payload": point.get("payload", {}),
            }

    def search_dense(self, collection, query_vector, k, filters=None):
        ids = []
        vectors = []
        payloads = []
        for pid, point in self._data.get(collection, {}).items():
            if not _matches(point["payload"], filters):
                continue
            ids.append(pid)
            vectors.append(np.asarray(point["vector"], dtype=np.float64))
            payloads.append(point["payload"])
        if not ids:
            return []

        matrix = np.stack(vectors)
        query = np.asarray(query_vector, dtype=np.float64)
        scores = matrix.dot(query)
        qnorm = np.linalg.norm(query)
        if qnorm > 0:
            scores = scores / qnorm
        order = np.argsort(-scores)
        out_ids = [ids[i] for i in order]
        out_scores = [float(scores[i]) for i in order]
        out_payloads = [payloads[i] for i in order]
        return self._result_list(list(zip(out_ids, out_scores, out_payloads))[:k])

    def search_sparse(self, collection, query_text, k, filters=None):
        # Real BM25 (BM25Okapi) over each collection's stored raw_text, per
        # Phase 4. We build the BM25 index AT QUERY TIME rather than caching
        # it on upsert: the store is used in tests with repeated/idempotent
        # re-upserts and deletes, so a cached index would go stale silently.
        # Query-time rebuild over the (small) in-memory collection is cheap
        # and always consistent with the current payloads. If there are no
        # matching points for the filter, return [] without attempting a BM25
        # build (an empty corpus would make BM25Okapi raise).
        docs = []
        ids = []
        for pid, point in self._data.get(collection, {}).items():
            if not _matches(point["payload"], filters):
                continue
            payload = point["payload"]
            text = (
                payload.get("raw_text")
                or payload.get("text")
                or payload.get("preferred_label")
                or ""
            )
            docs.append(text)
            ids.append(pid)
        if not docs:
            return []

        bm25 = BM25Okapi(_bm25_corpus_tokens(docs))
        scores = _bm25_scores(bm25, query_text)
        # Negative/zero BM25 scores mean "no term overlap" -> exclude them so
        # a query that does not match at all never shapes results.
        candidates = sorted(
            ((scores[i], ids[i]) for i in scores if scores[i] > 0),
            key=lambda x: -x[0],
        )
        payload_by_id = {
            pid: point["payload"]
            for pid, point in self._data.get(collection, {}).items()
        }
        return self._result_list(
            [(pid, score, payload_by_id[pid]) for score, pid in candidates[:k]]
        )

    def delete_by(self, collection, filters):
        coll = self._data.get(collection)
        if not coll:
            return
        to_delete = [pid for pid, point in coll.items() if _matches(point["payload"], filters)]
        for pid in to_delete:
            del coll[pid]

    def count(self, collection):
        return len(self._data.get(collection, {}))

    def health(self):
        return True


class QdrantStore(VectorStore):
    """Thin wrapper over qdrant-client. health() returns False rather than raising."""

    PAYLOAD_INDEX_FIELDS = ("candidate_id", "job_id", "section_type")

    def __init__(self, url=None, api_key=None, vector_size=None, timeout=10):
        if not _QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required for QdrantStore")
        self.url = url or "http://localhost:6333"
        self.vector_size = vector_size
        self.client = qdrant_client.QdrantClient(
            url=self.url, api_key=api_key, timeout=timeout
        )

    def _ensure_collection(self, collection):
        try:
            self.client.get_collection(collection_name=collection)
            return
        except Exception:
            pass
        if self.vector_size is None:
            raise RuntimeError(
                "QdrantStore requires vector_size to create collection %r" % collection
            )
        vectors_config = VectorParams(
            size=self.vector_size, distance=Distance.COSINE
        )
        self.client.create_collection(
            collection_name=collection, vectors_config=vectors_config
        )
        for field in self.PAYLOAD_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=collection, field_name=field, field_schema="keyword"
                )
            except Exception:
                pass

    def upsert(self, collection, points):
        self._ensure_collection(collection)
        self.client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point.get("payload", {}),
                )
                for point in points
            ],
        )

    def search_dense(self, collection, query_vector, k, filters=None):
        hits = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=k,
            query_filter=self._build_filter(filters),
        )
        return [
            {"id": hit.id, "score": hit.score, "payload": hit.payload}
            for hit in hits
        ]

    def search_sparse(self, collection, query_text, k, filters=None):
        # DEGRADED LEXICAL SCAN — NOT real BM25.
        #
        # Real BM25 is implemented for InMemoryStore (rank_bm25). Native
        # Qdrant sparse vectors would require a second named vector in the
        # collection schema at creation time, which changes the shared points
        # schema and cannot be validated without a live Qdrant instance
        # (none is available in this repository's environment). Until then,
        # QdrantStore keeps this token-overlap scan so the interface is
        # satisfied, but it is explicitly labeled degraded: any ablation
        # results produced against QdrantStore must be caveated as NOT using
        # real BM25.
        logger.warning(
            "QdrantStore.search_sparse using degraded lexical scan, not real BM25"
        )
        qtok = _tokenize(query_text)
        records, _ = self.client.scroll(
            collection_name=collection, limit=10000, with_payload=True, with_vectors=False
        )
        scored = []
        for point in records:
            if filters and not _matches(point.payload, filters):
                continue
            payload = point.payload
            hay = (
                payload.get("raw_text")
                or payload.get("text")
                or payload.get("preferred_label")
                or ""
            )
            overlap = len(qtok & _tokenize(hay))
            if overlap > 0:
                scored.append((overlap, point.id, payload))
        scored.sort(key=lambda x: -x[0])
        return self._result_list(scored[:k])

    def delete_by(self, collection, filters):
        self.client.delete(
            collection_name=collection,
            points_selector=FilterSelector(filter=self._build_filter(filters)),
        )

    def count(self, collection):
        return self.client.count(collection_name=collection).count

    def health(self):
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    @staticmethod
    def _build_filter(filters):
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            match_any = list(value) if isinstance(value, (list, tuple)) else [value]
            conditions.append(FieldCondition(key=key, match=MatchAny(any=match_any)))
        return Filter(must=conditions)
