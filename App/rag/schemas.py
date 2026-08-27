from typing import Literal, Optional

from pydantic import BaseModel


class ChunkMeta(BaseModel):
    candidate_id: str
    chunk_id: str
    section_type: str
    chunker_version: str
    embed_model: str


class Chunk(BaseModel):
    chunk_id: str
    candidate_id: str
    raw_text: str
    char_start: int
    char_end: int
    meta: ChunkMeta


class NormalizedRequirement(BaseModel):
    skill_name: str
    tier: Literal["required", "preferred"]
    source: Literal["llm_classifier", "keyword_fallback"]


class EvidenceSpan(BaseModel):
    chunk_id: str
    candidate_id: str
    text: str
    dense_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    fused_rank: Optional[int] = None
    cross_encoder_score: Optional[float] = None


class SkillVerdict(BaseModel):
    skill_name: str
    verdict: Literal["demonstrated", "claimed_only", "absent"]
    cited_chunk_ids: list[str]
    quote: Optional[str] = None


class CandidateEvidence(BaseModel):
    candidate_id: str
    verdicts: list[SkillVerdict]
    rejected_citations: int = 0


class ScoringArtifact(BaseModel):
    candidate_id: str
    scoring_version: str
    signal_breakdown: dict[str, float]
    final_score: float
