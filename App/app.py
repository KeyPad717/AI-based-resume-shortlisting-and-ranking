from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import shutil
import tempfile
import json
import re
import uuid
from pathlib import Path

from pipeline import (
    process_resumes,
    get_rag_stores,
    _ensure_indexed,
    _rag_runtime_cache,
    score_candidate_rag,
    extract_text,
    fix_numbers,
    clean_text,
    validate_extracted_struct,
    call_llm_extraction,
    experience_score,
    education_score,
    project_score,
    normalize_weights,
    generate_explanation,
    embedding_model,
)

from rag.api_models import (
    JobCreateResponse,
    NormalizedRequirementOut,
    ResumeIngestStatus,
    JobStatusResponse,
    ScoreRequest,
    ScoreResult,
    SkillVerdictOut,
    EvidenceDetailResponse,
    SpanOut,
)

app = FastAPI(title="Resume ATS Pipeline API")

# Allow Frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Phase 7: in-memory job store.
#
# Deliberately an in-memory dict keyed by uuid4, consistent with this project's
# in-memory-first pattern (InMemoryStore is the default RAG store; no SQLite /
# external persistence exists anywhere yet). Job state (jd struct, normalized
# requirements, per-resume ingestion status) is lost on process restart — that
# is a Phase 10 (productionization) concern, NOT a Phase 7 gap.
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}

# Process-lifetime temp root for uploaded resume files. Background ingestion
# tasks run AFTER the request's TemporaryDirectory would have been cleaned, so
# resume bytes are parked here (not in a per-request temp dir) until indexed.
_JOB_TMPROOT = tempfile.mkdtemp(prefix="ats_jobs_")


def _candidate_id_from_file(file_path: str) -> str:
    """Content-hash candidate_id for a resume without indexing it.

    Reuses Phase 2's ResumeIngestor (cheap: text extraction + chunking, no
    embedding) so the /resumes endpoint can dedupe by content synchronously.
    """
    from rag.ingest import ResumeIngestor

    _chunks, report = ResumeIngestor().ingest(file_path)
    return report.candidate_id


def _candidate_name(r_struct: dict, file_path: str) -> str:
    """Name derivation mirroring process_resumes (filename as fallback)."""
    filename_name = re.sub(
        r"\b(mt|bt|cv|resume|updated|final|copy|sde)\b", " ",
        Path(file_path).stem, flags=re.IGNORECASE,
    )
    filename_name = re.sub(r"[^a-zA-Z\s]", "", filename_name).strip().title()
    return r_struct["name"] if r_struct["name"] else filename_name


def _ingest_resume(job_id: str, candidate_id: str, file_path: str, filename: str):
    """Background task: parse + index one resume, updating ingestion status.

    Runs after the POST /resumes response has been sent (FastAPI BackgroundTasks).
    Any failure marks the status "error" rather than crashing the request.
    """
    job = _jobs.get(job_id)
    if job is None:
        return
    status = job["resumes"].get(candidate_id)
    if status is None:
        return
    status["status"] = "processing"
    try:
        r_raw = extract_text(file_path)
        r_text = fix_numbers(clean_text(r_raw))
        r_struct = validate_extracted_struct(
            call_llm_extraction(r_text, is_jd=False), r_text, is_jd=False
        )
        rt = get_rag_stores()
        got_cid = _ensure_indexed(file_path, rt["resume_store"], rt["embedder"])
        status["candidate_id"] = got_cid
        status["r_text"] = r_text
        status["r_struct"] = r_struct
        status["cand_name"] = _candidate_name(r_struct, file_path)
        status["status"] = "done"
        status["error"] = None
    except Exception as e:
        status["status"] = "error"
        status["error"] = str(e)


def _assemble_score(job: dict, status: dict, rag_out: dict, user_weights=None) -> dict:
    """Build one ScoreResult dict for a candidate from the RAG signals + legacy
    experience/education/project signals (mirrors process_resumes' arithmetic)."""
    req_score = rag_out["required_skills"]
    matched = rag_out["matched"]
    missing = rag_out["missing"]
    sem_scr = rag_out["semantic"]
    rerank_scr = rag_out["reranker"]
    ev_scr = rag_out["evidence"]

    weights = rag_out["weights"]
    if user_weights:
        weights = normalize_weights(
            {k: max(0, float(v)) / 100.0 for k, v in user_weights.items()}
        )

    r_struct = status["r_struct"]
    r_text = status["r_text"]
    jd_struct = job["jd_struct"]

    exp_scr = experience_score(jd_struct.get("experience_years", 0),
                               r_struct.get("experience_years", 0))
    edu_scr = education_score(jd_struct.get("education", []),
                              r_struct.get("education", []), r_text)
    proj_scr = project_score(job["jd_text"], r_struct.get("projects", []),
                             embedding_model)

    final = sum([
        weights.get('required_skills', 0) * req_score,
        weights.get('semantic', 0) * sem_scr,
        weights.get('reranker', 0) * rerank_scr,
        weights.get('experience', 0) * exp_scr,
        weights.get('education', 0) * edu_scr,
        weights.get('projects', 0) * proj_scr,
        weights.get('evidence', 0) * ev_scr,
    ])

    res = {
        "name": status.get("cand_name", ""),
        "final_score": round(float(final), 4),
        "matched_skills": matched,
        "missing_skills": missing,
        "experience_years": r_struct.get("experience_years", 0),
        "required_skill_score": round(req_score, 2),
        "semantic_score": round(sem_scr, 2),
        "reranker_score": round(rerank_scr, 2),
        "education_score": round(edu_scr, 2),
        "project_score": round(proj_scr, 2),
        "semantic_summary": r_struct.get("semantic_summary", ""),
        "experience_breakdown": r_struct.get("experience_breakdown", []),
        "projects": r_struct.get("projects", []),
        "education": r_struct.get("education", []),
        "scoring_version": "v1-rag",
        "skill_verdicts": [
            {
                "skill_name": v.skill_name,
                "verdict": v.verdict,
                "cited_chunk_ids": v.cited_chunk_ids,
                "quote": v.quote,
            }
            for v in rag_out["verdicts"]
        ],
    }
    res["explanation"] = generate_explanation(res)
    return res


def _evidence_detail(job: dict, candidate_id: str, skill: Optional[str]):
    """Retrieve + verify evidence for a candidate, returning a list of
    EvidenceDetailResponse (one per skill, or a single skill when filtered)."""
    reqs = job["normalized_requirements"]
    if skill is not None:
        reqs = [r for r in reqs if r.skill_name == skill]
        if not reqs:
            raise HTTPException(
                status_code=404,
                detail=f"skill '{skill}' not found in job requirements",
            )

    from rag.ontology import SkillNormalizer
    from rag.retrieval import HybridRetriever, RRFFuser, SkillEvidenceRetriever
    from rag.evidence import CitationValidator, EvidenceVerifier

    rt = get_rag_stores()
    normalizer = SkillNormalizer(rt["ontology_store"], rt["embedder"], rt["llm_client"])
    skill_retriever = SkillEvidenceRetriever(
        HybridRetriever(rt["resume_store"], rt["embedder"], RRFFuser()),
        rt["reranker"],
    )

    spans_by_skill = {}
    for r in reqs:
        alt = normalizer.alt_labels_for(r.skill_name)
        spans_by_skill[r.skill_name] = skill_retriever.for_requirement(
            r, candidate_id, alt_labels=alt
        )

    from pipeline import _CachedEvidenceRetriever
    verifier = EvidenceVerifier(
        _CachedEvidenceRetriever(spans_by_skill),
        rt["llm_client"],
        CitationValidator(),
        alt_labels_provider=normalizer.alt_labels_for,
    )
    evidence = verifier.verify(candidate_id, reqs)
    verdicts = {v.skill_name: v for v in evidence.verdicts}

    out = []
    for r in reqs:
        v = verdicts.get(r.skill_name)
        out.append(EvidenceDetailResponse(
            candidate_id=candidate_id,
            skill_name=r.skill_name,
            verdict=v.verdict if v else "absent",
            cited_chunk_ids=list(v.cited_chunk_ids) if v else [],
            quote=v.quote if v else None,
            retrieved_spans=[
                SpanOut(
                    chunk_id=s.chunk_id,
                    text=s.text,
                    dense_rank=s.dense_rank,
                    lexical_rank=s.lexical_rank,
                    fused_rank=s.fused_rank,
                    cross_encoder_score=s.cross_encoder_score,
                )
                for s in spans_by_skill.get(r.skill_name, [])
            ],
        ))
    return out


@app.post("/api/score")
async def score_candidates(
    jd_file: UploadFile = File(...),
    resume_files: List[UploadFile] = File(...),
    weights: Optional[str] = Form(None)
):
    # Parse weights if provided
    user_weights = None
    if weights:
        try:
            user_weights = json.loads(weights)
        except Exception:
            pass

    if not jd_file.filename:
        raise HTTPException(status_code=400, detail="JD File missing")
    if not resume_files or len(resume_files) == 0:
        raise HTTPException(status_code=400, detail="Resume files missing")

    allowed_exts = {".pdf", ".docx"}
    if not os.path.splitext(jd_file.filename)[1].lower() in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported JD file type: {jd_file.filename}. Only PDF and DOCX are accepted.")
    for r_file in resume_files:
        if r_file.filename and os.path.splitext(r_file.filename)[1].lower() not in allowed_exts:
            raise HTTPException(status_code=400, detail=f"Unsupported resume file type: {r_file.filename}. Only PDF and DOCX are accepted.")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save JD
        jd_path = os.path.join(tmpdir, os.path.basename(jd_file.filename))
        with open(jd_path, "wb") as f:
            shutil.copyfileobj(jd_file.file, f)
            
        # Save Resumes
        resume_paths = []
        for r_file in resume_files:
            if not r_file.filename: continue
            path = os.path.join(tmpdir, os.path.basename(r_file.filename))
            with open(path, "wb") as f:
                shutil.copyfileobj(r_file.file, f)
            resume_paths.append(path)
            
        # Run Pipeline
        try:
            results = process_resumes(jd_path, resume_paths, user_weights=user_weights)
            return {"status": "success", "data": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Phase 7: job-based surface (image ingestion from scoring).
# These endpoints ALWAYS use the RAG-scored path (equivalent to rag_enabled=True)
# by design — splitting ingestion from scoring is the entire point of this API.
# It is intentionally NOT configurable in this phase.
# ---------------------------------------------------------------------------

@app.post("/api/jobs", response_model=JobCreateResponse)
async def create_job(
    jd_file: UploadFile = File(...),
    weights: Optional[str] = Form(None),
):
    if not jd_file.filename:
        raise HTTPException(status_code=400, detail="JD File missing")
    allowed_exts = {".pdf", ".docx"}
    if os.path.splitext(jd_file.filename)[1].lower() not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported JD file type: {jd_file.filename}. Only PDF and DOCX are accepted.")

    user_weights = None
    if weights:
        try:
            user_weights = json.loads(weights)
        except Exception:
            pass

    job_id = str(uuid.uuid4())

    # Extract + classify the JD ONCE here (not on every score call).
    with tempfile.TemporaryDirectory() as tmpdir:
        jd_path = os.path.join(tmpdir, os.path.basename(jd_file.filename))
        with open(jd_path, "wb") as f:
            shutil.copyfileobj(jd_file.file, f)

        jd_raw = extract_text(jd_path)
        jd_text = fix_numbers(clean_text(jd_raw))
        jd_struct = validate_extracted_struct(
            call_llm_extraction(jd_text, is_jd=True), jd_text, is_jd=True
        )

    rt = get_rag_stores()
    from rag.ontology import SkillNormalizer
    normalizer = SkillNormalizer(rt["ontology_store"], rt["embedder"], rt["llm_client"])
    requirements = normalizer.normalize(jd_text, jd_struct["skills"])

    _jobs[job_id] = {
        "job_id": job_id,
        "jd_text": jd_text,
        "jd_struct": jd_struct,
        "normalized_requirements": requirements,
        "default_weights": user_weights,
        "resumes": {},
    }

    return JobCreateResponse(
        job_id=job_id,
        normalized_requirements=[
            NormalizedRequirementOut(
                skill_name=r.skill_name, tier=r.tier, source=r.source
            )
            for r in requirements
        ],
    )


@app.post("/api/jobs/{job_id}/resumes", response_model=JobStatusResponse)
async def upload_resumes(
    job_id: str,
    resume_files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    if not resume_files or len(resume_files) == 0:
        raise HTTPException(status_code=400, detail="Resume files missing")

    allowed_exts = {".pdf", ".docx"}
    job_dir = os.path.join(_JOB_TMPROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)

    added = []
    for r_file in resume_files:
        if not r_file.filename:
            continue
        if os.path.splitext(r_file.filename)[1].lower() not in allowed_exts:
            raise HTTPException(status_code=400, detail=f"Unsupported resume file type: {r_file.filename}. Only PDF and DOCX are accepted.")

        saved_path = os.path.join(job_dir, os.path.basename(r_file.filename))
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(r_file.file, f)

        candidate_id = _candidate_id_from_file(saved_path)
        if not candidate_id:
            raise HTTPException(status_code=400, detail=f"Could not extract any content from {r_file.filename}")

        # Dedup by content hash: the same resume uploaded twice resolves to one
        # candidate_id and does not create a duplicate entry.
        if candidate_id in job["resumes"]:
            continue

        job["resumes"][candidate_id] = {
            "candidate_id": candidate_id,
            "filename": os.path.basename(r_file.filename),
            "status": "pending",
            "error": None,
            "file_path": saved_path,
            "r_text": None,
            "r_struct": None,
            "cand_name": "",
        }
        added.append(candidate_id)
        # Ingest asynchronously; the request returns with "pending" status.
        background_tasks.add_task(
            _ingest_resume, job_id, candidate_id, saved_path, r_file.filename
        )

    return JobStatusResponse(
        job_id=job_id,
        resumes=[_to_status(s) for s in job["resumes"].values()],
    )


def _to_status(status: dict) -> ResumeIngestStatus:
    return ResumeIngestStatus(
        candidate_id=status.get("candidate_id", ""),
        filename=status.get("filename", ""),
        status=status.get("status", "pending"),
        error=status.get("error"),
    )


@app.get("/api/jobs/{job_id}/status", response_model=JobStatusResponse)
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(
        job_id=job_id,
        resumes=[_to_status(s) for s in job["resumes"].values()],
    )


@app.post("/api/jobs/{job_id}/score", response_model=List[ScoreResult])
async def score_job(job_id: str, req: Optional[ScoreRequest] = None):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    done = {
        cid: st for cid, st in job["resumes"].items() if st["status"] == "done"
    }
    if not done:
        raise HTTPException(
            status_code=409,
            detail="No indexed resumes yet for this job; check /status",
        )

    user_weights = (req.weights if req else None) or job.get("default_weights")
    rt = get_rag_stores()
    results = []
    for cid, st in done.items():
        rag_out = score_candidate_rag(
            job["jd_text"], st["r_text"], st["r_struct"], job["jd_struct"],
            st["file_path"],
            rt["ontology_store"], rt["resume_store"], rt["embedder"],
            rt["reranker"], rt["llm_client"],
            requirements=job["normalized_requirements"],
            candidate_id=cid,
        )
        results.append(_assemble_score(job, st, rag_out, user_weights))

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return [ScoreResult(**r) for r in results]


@app.get("/api/jobs/{job_id}/candidates/{candidate_id}/evidence",
         response_model=List[EvidenceDetailResponse])
async def candidate_evidence(
    job_id: str,
    candidate_id: str,
    skill: Optional[str] = Query(None),
):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if candidate_id not in job["resumes"]:
        raise HTTPException(status_code=404, detail="candidate not found")
    return _evidence_detail(job, candidate_id, skill)


@app.get("/api/health")
async def health():
    # Liveness + whether the RAG runtime has been initialized yet. Does NOT
    # call get_rag_stores() — /health must not force lazy initialization.
    return {
        "status": "ok",
        "rag_initialized": bool(_rag_runtime_cache),
    }


@app.post("/api/warmup")
async def warmup():
    # Force lazy RAG runtime + embedder model load so the first real request
    # is not slow. Returns once ready.
    rt = get_rag_stores()
    rt["embedder"].warmup()
    return {"status": "ready"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
