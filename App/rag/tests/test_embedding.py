import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.embedding import EmbeddingService

TEXTS = [
    "Developed supervised machine learning models for forest cover classification",
    "Built a role-based access control system with JWT authorization",
    "Orchestrated Kubernetes clusters for microservices deployment",
    "Designed RESTful APIs for profile search and alumni data management",
    "Used SQL and MySQL for relational database queries",
]


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def test_batch_encode_dimension():
    svc = EmbeddingService.get_instance()
    got = svc.encode(TEXTS)
    assert len(got) == 5, "expected 5 vectors for a 5-text batch"
    dim = svc.get_dimension()
    assert all(len(v) == dim for v in got), "every vector must match the model dimension"
    print("ENCODE OK: batch of 5 texts -> 5 vectors, each dimension=%d" % dim)


def test_determinism():
    svc = EmbeddingService.get_instance()
    a = svc.encode(["running the same text twice"])
    b = svc.encode(["running the same text twice"])
    assert a[0] == b[0], "encoding the same text must be deterministic"
    print("DETERMINISM OK: identical text encodes identically")


def test_unit_normalized():
    svc = EmbeddingService.get_instance()
    for text in TEXTS:
        v = svc.encode([text])[0]
        n = _norm(v)
        assert abs(n - 1.0) < 1e-3, "vector must be unit-normalized, got norm %.4f" % n
    print("NORMALIZATION OK: all vectors unit-normalized (norm ~1.0)")


def test_warmup_before_encode():
    fresh = EmbeddingService()
    t0 = time.time()
    fresh.warmup()
    # warmup must not raise and must load the model
    fresh.warmup()
    print("WARMUP OK: warmup() callable before encode (load took %.1fs)" % (time.time() - t0))


if __name__ == "__main__":
    test_batch_encode_dimension()
    test_determinism()
    test_unit_normalized()
    test_warmup_before_encode()
    print("ALL EMBEDDING TESTS PASSED")
