"""Phase 7: job-based FastAPI surface integration tests.

Drives the new /api/jobs* endpoints against the app with deterministic fakes
(no live LLM / Qdrant / real embedder) so they run fast and repeatably. The
FastAPI TestClient executes BackgroundTasks synchronously, so POST /resumes
returns with ingestion already "done" — no sleep/polling is needed (documented
choice).

Test 5 (legacy /api/score contract) patches ``app.process_resumes`` to return the
Phase 6 golden baseline results and asserts the endpoint surfaces them
byte-for-byte. The process-level equivalence of process_resumes(rag off) to that
golden file is proven separately in test_scoring_integration.py
(test_rag_disabled_byte_identical_to_golden_baseline); here we isolate and prove
the *HTTP contract* did not move.
"""

import hashlib
import json
import math
import os
import re
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import app
from rag.config import RagSettings
from rag.ontology import SkillNormalizer
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
# Deterministic fakes (no real models / LLM / Qdrant).
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
    def encode(self, texts):
        return [_bow(t) for t in texts]


class FakeCrossEncoder:
    def __init__(self, logits, default=1.5):
        self._logits = list(logits)
        self._default = default
        self._ptr = 0

    def predict(self, pairs):
        out = []
        for _ in range(len(pairs)):
            if self._ptr < len(self._logits):
                out.append(self._logits[self._ptr])
                self._ptr += 1
            else:
                out.append(self._default)
        return out


class FakeOntologyStore:
    def __init__(self, concepts):
        self._concepts = concepts

    def search_dense(self, collection, query_vector, k, filters=None):
        best, best_score = None, -1.0
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
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def _evidence_verdicts(self, content):
        verdicts = []
        for line in content.splitlines():
            line = line.strip()
            if "SKILL:" in line:
                skill = line.split("SKILL:", 1)[1].strip()
                verdicts.append({
                    "skill_name": skill, "verdict": "claimed_only",
                    "cited_chunk_ids": [], "quote": None,
                })
        return verdicts


def _runtime(logits=None, classification=None):
    return {
        "ontology_store": FakeOntologyStore({
            "kubernetes": {"preferred": "Kubernetes", "alt": ["k8s"], "score": 0.3},
            "docker": {"preferred": "Docker", "alt": ["containerization"], "score": 0.3},
        }),
        "resume_store": InMemoryStore(),
        "embedder": FakeEmbedder(),
        "reranker": CrossEncoderReranker(
            model=FakeCrossEncoder(logits if logits is not None else [1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
        ),
        "llm_client": FakeScriptedLLM(
            classification if classification is not None else
            {"required": ["Kubernetes", "Docker", "Python"], "preferred": []}
        ),
    }


def _stub_llm(text, is_jd=False, retries=2):
    return {
        "candidate_name": "" if is_jd else "Test Candidate",
        "skills": ["python", "kubernetes", "docker"],
        "experience": [{"role": "sde", "type": "full-time", "duration_years": 3}],
        "education": ["BTech"],
        "projects": ["k8s platform"],
        "semantic_summary": "test summary",
    }


def _jd_files():
    path = os.path.join(JD_DIR, "JD.pdf")
    with open(path, "rb") as fh:
        return [("jd_file", ("JD.pdf", fh.read(), "application/pdf"))]


def _resume_files(paths=None):
    files = []
    if paths is None:
        paths = [os.path.join(RESUMES_DIR, f) for f in json.load(open(GOLDEN))["resumes"]]
    for p in paths:
        with open(p, "rb") as fh:
            files.append(("resume_files", (os.path.basename(p), fh.read(), "application/pdf")))
    return files


from contextlib import ExitStack


class _PatchStack:
    """Context manager that enters several mock patches together."""

    def __init__(self, patchers):
        self._patchers = patchers

    def __enter__(self):
        self._stack = ExitStack()
        for p in self._patchers:
            self._stack.enter_context(p)
        return self

    def __exit__(self, *exc):
        self._stack.close()


def _base_patches(runtime):
    return _PatchStack([
        patch.object(app, "get_rag_stores", return_value=runtime),
        patch.object(app, "call_llm_extraction", side_effect=_stub_llm),
    ])


# ---------------------------------------------------------------------------
# 1. Full happy path.
# ---------------------------------------------------------------------------

def test_1_full_happy_path():
    from fastapi.testclient import TestClient
    client = TestClient(app.app)
    rt = _runtime()
    with _base_patches(rt):
        # Create a job from the real JD.
        r = client.post("/api/jobs", files=_jd_files())
        assert r.status_code == 200, r.text
        body = r.json()
        job_id = body["job_id"]
        assert job_id
        assert len(body["normalized_requirements"]) > 0
        assert all(req["skill_name"] and req["tier"] in ("required", "preferred")
                   for req in body["normalized_requirements"])

        # Upload the 7 real sample resumes.
        r = client.post(f"/api/jobs/{job_id}/resumes", files=_resume_files())
        assert r.status_code == 200, r.text
        statuses = r.json()["resumes"]
        assert len(statuses) == 7

        # BackgroundTasks run with the TestClient, but ingestion is async and
        # the immediate POST response reflects "pending". Poll /status (bounded)
        # until every resume is "done".
        done = False
        for _ in range(40):
            r = client.get(f"/api/jobs/{job_id}/status")
            assert r.status_code == 200
            statuses = r.json()["resumes"]
            if all(s["status"] == "done" for s in statuses):
                done = True
                break
        assert done, statuses

        # Score -> 7 ScoreResults, sorted descending, all v1-rag.
        r = client.post(f"/api/jobs/{job_id}/score")
        assert r.status_code == 200, r.text
        results = r.json()
        assert len(results) == 7
        assert all(res["scoring_version"] == "v1-rag" for res in results)
        finals = [res["final_score"] for res in results]
        assert finals == sorted(finals, reverse=True)

        # Evidence-detail for the top candidate's first skill returns spans.
        top = results[0]
        cid = None
        for s in statuses:
            if s["candidate_id"]:
                cid = s["candidate_id"]
                break
        assert cid
        first_skill = top["skill_verdicts"][0]["skill_name"]
        r = client.get(
            f"/api/jobs/{job_id}/candidates/{cid}/evidence",
            params={"skill": first_skill},
        )
        assert r.status_code == 200, r.text
        ev = r.json()
        assert isinstance(ev, list) and len(ev) == 1
        assert ev[0]["skill_name"] == first_skill
        assert "verdict" in ev[0] and "retrieved_spans" in ev[0]


# ---------------------------------------------------------------------------
# 2. Duplicate resume upload resolves to one candidate_id.
# ---------------------------------------------------------------------------

def test_2_duplicate_upload_dedupes():
    from fastapi.testclient import TestClient
    client = TestClient(app.app)
    rt = _runtime()
    one = os.path.join(RESUMES_DIR, json.load(open(GOLDEN))["resumes"][0])
    with _base_patches(rt):
        job_id = client.post("/api/jobs", files=_jd_files()).json()["job_id"]

        r1 = client.post(f"/api/jobs/{job_id}/resumes", files=_resume_files([one, one]))
        assert r1.status_code == 200, r1.text
        status1 = r1.json()["resumes"]
        assert len(status1) == 1, status1

        r2 = client.post(f"/api/jobs/{job_id}/resumes", files=_resume_files([one]))
        status2 = r2.json()["resumes"]
        # Still just ONE entry for that candidate_id, not two.
        assert len(status2) == 1, status2
        assert status2[0]["candidate_id"] == status1[0]["candidate_id"]


# ---------------------------------------------------------------------------
# 3. Score before any resume ingested -> 409 with actionable message.
# ---------------------------------------------------------------------------

def test_3_score_before_ingestion_returns_409():
    from fastapi.testclient import TestClient
    client = TestClient(app.app)
    rt = _runtime()
    with _base_patches(rt):
        job_id = client.post("/api/jobs", files=_jd_files()).json()["job_id"]
        r = client.post(f"/api/jobs/{job_id}/score")
        assert r.status_code == 409, r.status_code
        detail = r.json().get("detail", "")
        assert "No indexed resumes" in detail
        assert "/status" in detail


# ---------------------------------------------------------------------------
# 4. 404 handling.
# ---------------------------------------------------------------------------

def test_4_404_handling():
    from fastapi.testclient import TestClient
    client = TestClient(app.app)
    rt = _runtime()
    with _base_patches(rt):
        # Nonexistent job -> 404.
        r = client.get(f"/api/jobs/{uuid.uuid4()}/status")
        assert r.status_code == 404

        # Existing job, nonexistent candidate -> 404.
        job_id = client.post("/api/jobs", files=_jd_files()).json()["job_id"]
        r = client.get(f"/api/jobs/{job_id}/candidates/nonexistent-candidate/evidence")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 5. Legacy /api/score contract unchanged (vs Phase 6 golden baseline).
# ---------------------------------------------------------------------------

def test_5_legacy_score_contract_unchanged():
    from fastapi.testclient import TestClient
    client = TestClient(app.app)
    baseline = json.load(open(GOLDEN))
    golden_results = baseline["results"]
    resume_files = [os.path.join(RESUMES_DIR, f) for f in baseline["resumes"]]

    # The HTTP endpoint is a pure passthrough of process_resumes' output. Patch
    # process_resumes to return the Phase 6 golden results and assert the
    # endpoint surfaces them byte-for-byte (proving the request/response
    # contract did not move). The process-level equivalence of process_resumes
    # to this golden file is proven separately in test_scoring_integration.py.
    with patch.object(app, "process_resumes", return_value=golden_results), \
         patch.dict(os.environ, {"RAG_ENABLED": "false"}):
        r = client.post(
            "/api/score",
            files=_jd_files() + _resume_files(resume_files),
        )
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data == golden_results, "POST /api/score response moved from golden baseline"
    # No extra/missing keys at the top level.
    assert set(r.json().keys()) == {"status", "data"}


# ---------------------------------------------------------------------------
# 6. /score does not re-normalize the JD or re-ingest resumes.
# ---------------------------------------------------------------------------

def test_6_score_does_not_reparse_or_reingest():
    from fastapi.testclient import TestClient
    client = TestClient(app.app)
    rt = _runtime()

    with patch.object(app, "get_rag_stores", return_value=rt), \
         patch.object(app, "call_llm_extraction", side_effect=_stub_llm):

        norm_calls = {"n": 0}
        orig_normalize = SkillNormalizer.normalize

        def _normalize_spy(self, jd_text, extracted_skills):
            norm_calls["n"] += 1
            return orig_normalize(self, jd_text, extracted_skills)

        indexed_calls = {"n": 0}
        orig_ensure = app._ensure_indexed

        def _ensure_spy(r_path, resume_store, embedder):
            indexed_calls["n"] += 1
            return orig_ensure(r_path, resume_store, embedder)

        with patch.object(SkillNormalizer, "normalize", _normalize_spy), \
             patch.object(app, "_ensure_indexed", _ensure_spy):

            job_id = client.post("/api/jobs", files=_jd_files()).json()["job_id"]
            # normalize called exactly once (at job creation).
            assert norm_calls["n"] == 1

            one = os.path.join(RESUMES_DIR, json.load(open(GOLDEN))["resumes"][0])
            client.post(f"/api/jobs/{job_id}/resumes", files=_resume_files([one]))

            idx_before = indexed_calls["n"]
            norm_before = norm_calls["n"]

            r1 = client.post(f"/api/jobs/{job_id}/score")
            assert r1.status_code == 200, r1.text
            r2 = client.post(f"/api/jobs/{job_id}/score")
            assert r2.status_code == 200, r2.text

            # Scoring must NOT re-classify the JD nor re-ingest resumes.
            assert norm_calls["n"] == norm_before, "normalize called again during score"
            assert indexed_calls["n"] == idx_before, "_ensure_indexed called again during score"
            assert r1.json() == r2.json()


# ---------------------------------------------------------------------------
# 7. /health is lazy; /warmup forces RAG init.
# ---------------------------------------------------------------------------

def test_7_health_lazy_and_warmup_forces():
    from fastapi.testclient import TestClient
    client = TestClient(app.app)

    # /health must not call get_rag_stores / EmbeddingService.
    get_rag_calls = {"n": 0}
    orig_get = app.get_rag_stores

    def _get_spy():
        get_rag_calls["n"] += 1
        return orig_get()

    warmed = {"n": 0}
    rt = _runtime()

    class WarmFakeEmbedder:
        def warmup(self):
            warmed["n"] += 1

    rt["embedder"] = WarmFakeEmbedder()

    rt_factory = {"n": 0}

    def _rt_spy():
        rt_factory["n"] += 1
        return rt

    with patch.object(app, "get_rag_stores", _rt_spy):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        # Health did NOT trigger runtime init.
        assert rt_factory["n"] == 0
        assert warmed["n"] == 0

        r = client.post("/api/warmup")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"
        # Warmup forced runtime init + embedder warmup.
        assert rt_factory["n"] >= 1
        assert warmed["n"] >= 1


if __name__ == "__main__":
    client_mods = []
    test_1_full_happy_path()
    test_2_duplicate_upload_dedupes()
    test_3_score_before_ingestion_returns_409()
    test_4_404_handling()
    test_5_legacy_score_contract_unchanged()
    test_6_score_does_not_reparse_or_reingest()
    test_7_health_lazy_and_warmup_forces()
    print("ALL API JOBS TESTS PASSED")
