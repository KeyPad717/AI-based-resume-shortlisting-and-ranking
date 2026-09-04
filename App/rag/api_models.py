"""Phase 7: API-contract models for the job-based FastAPI surface.

This is a separate layer from ``rag/schemas.py`` (Phase 1's internal data-layer
schemas). These models describe the HTTP request/response contract only. The
legacy ``POST /api/score`` endpoint keeps its raw-dict contract untouched; this
module adds models purely for the new ``/api/jobs*`` endpoints.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class JobCreateRequest(BaseModel):
    """Optional weight overrides for a job.

    The JD file itself arrives via multipart, so this model only carries the
    optional weight overrides — mirroring the legacy ``/api/score`` endpoint's
    ``weights`` Form field (a JSON string parsed into this dict). If absent, the
    RAG default weights are used at score time.
    """

    weights: Optional[dict] = None


class NormalizedRequirementOut(BaseModel):
    """Serialized form of ``rag.schemas.NormalizedRequirement``."""

    skill_name: str
    tier: Literal["required", "preferred"]
    source: Literal["llm_classifier", "keyword_fallback"]


class JobCreateResponse(BaseModel):
    job_id: str
    normalized_requirements: list[NormalizedRequirementOut]


class ResumeIngestStatus(BaseModel):
    candidate_id: str
    filename: str
    status: Literal["pending", "processing", "done", "error"]
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    resumes: list[ResumeIngestStatus]


class ScoreRequest(BaseModel):
    """Optional weight overrides for ``POST /api/jobs/{id}/score``."""

    weights: Optional[dict] = None


class SkillVerdictOut(BaseModel):
    skill_name: str
    verdict: Literal["demonstrated", "claimed_only", "absent"]
    cited_chunk_ids: list[str]
    quote: Optional[str] = None


class ScoreResult(BaseModel):
    """One candidate's score — a Pydantic wrap of the per-candidate dict shape
    process_resumes already returns (name, final_score, ...). Field names are
    intentionally identical to that shape; nothing is dropped or renamed."""

    name: str
    final_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    experience_years: float
    required_skill_score: float
    semantic_score: float
    reranker_score: float
    education_score: float
    project_score: float
    semantic_summary: str
    experience_breakdown: list
    projects: list
    education: list
    scoring_version: str
    skill_verdicts: Optional[list[SkillVerdictOut]] = None
    explanation: str


class SpanOut(BaseModel):
    chunk_id: str
    text: str
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    fused_rank: Optional[float] = None
    cross_encoder_score: Optional[float] = None


class EvidenceDetailResponse(BaseModel):
    candidate_id: str
    skill_name: str
    verdict: Literal["demonstrated", "claimed_only", "absent"]
    cited_chunk_ids: list[str]
    quote: Optional[str] = None
    retrieved_spans: list[SpanOut]
