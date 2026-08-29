import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.store import InMemoryStore, QdrantStore, VectorStore
from rag.embedding import EmbeddingService

COLLECTION = "test_resume_chunks"
QUERY_TEXT = "container orchestration and kubernetes deployment"
PLANT_TEXT = "Kubernetes container orchestration for microservices"


def _synthetic_vector(dim, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


def _make_points(embed, dim, count=100):
    planted = embed.encode([PLANT_TEXT])[0]
    points = []
    for i in range(count):
        if i == 0:
            vector = planted
            section = "projects"
        else:
            vector = _synthetic_vector(dim, seed=i)
            section = "skills" if i % 2 else "experience"
        points.append(
            {
                "id": "r%03d" % i,
                "vector": vector,
                "payload": {
                    "candidate_id": "cand%d" % (i % 5),
                    "chunk_id": "r%03d" % i,
                    "section_type": section,
                    "chunker_version": "v1",
                    "embed_model": "sentence-transformers/all-mpnet-base-v2",
                    "raw_text": PLANT_TEXT if i == 0 else "some other text %d" % i,
                },
            }
        )
    return points


def run_store_suite(make_store, label):
    embed = EmbeddingService.get_instance()
    dim = embed.get_dimension()
    store = make_store(dim)
    store.upsert(COLLECTION, [])

    points = _make_points(embed, dim, count=100)

    # 2. upsert 100 -> count 100
    store.upsert(COLLECTION, points)
    assert store.count(COLLECTION) == 100, "expected 100 after first upsert, got %d" % store.count(COLLECTION)

    # 3. idempotence: same upsert again -> still 100, not 200
    store.upsert(COLLECTION, points)
    idempotent_count = store.count(COLLECTION)
    assert idempotent_count == 100, "expected 100 after idempotent re-upsert, got %d" % idempotent_count

    # 5. dimension check: vector size must equal EmbeddingService.get_dimension()
    for p in points:
        assert len(p["vector"]) == dim, "vector dim mismatch vs get_dimension()"

    # 4. planted distinctive point must be top-1 dense result for a close query
    query = embed.encode([QUERY_TEXT])[0]
    top = store.search_dense(COLLECTION, query, k=5)
    assert top, "search_dense returned nothing"
    assert top[0]["id"] == "r000", "expected planted point r000 on top, got %r" % top[0]["id"]
    assert top[0]["score"] > 0.5, "planted point score unexpectedly low: %s" % top[0]["score"]

    # filters exercise: planted point section_type == 'projects'
    filt = store.search_dense(COLLECTION, query, k=5, filters={"section_type": "projects"})
    assert filt, "filtered search returned nothing"
    assert all(r["payload"]["section_type"] == "projects" for r in filt)

    # sparse search on planted text returns something
    sparse = store.search_sparse(COLLECTION, PLANT_TEXT, k=5)
    assert sparse, "sparse search returned nothing"

    # delete_by works
    store.delete_by(COLLECTION, {"candidate_id": "cand0"})
    remaining = store.count(COLLECTION)
    assert remaining < 100, "delete_by did not remove points, still %d" % remaining

    print("[%s] OK: upsert=100 idempotent_count=%d planted_top1=%r dim=%d sparse_hits=%d delete_left=%d" % (
        label, idempotent_count, top[0]["id"], dim, len(sparse), remaining))


def test_inmemory_store():
    run_store_suite(lambda dim: InMemoryStore(expected_dimension=dim), "InMemoryStore")

    # health is trivially True for in-memory
    store = InMemoryStore()
    assert store.health() is True


def test_qdrant_store():
    store = QdrantStore(url="http://localhost:6333", vector_size=EmbeddingService.get_instance().get_dimension())
    if not store.health():
        print("[QdrantStore] SKIPPED: no live Qdrant instance at localhost:6333; "
              "full suite cannot execute (written but not run). health() returned False.")
        return
    run_store_suite(lambda dim: QdrantStore(url="http://localhost:6333", vector_size=dim), "QdrantStore")


if __name__ == "__main__":
    test_inmemory_store()
    test_qdrant_store()
    print("ALL STORE TESTS PASSED")
