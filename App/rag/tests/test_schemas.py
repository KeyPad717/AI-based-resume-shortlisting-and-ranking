import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.schemas import (
    CandidateEvidence,
    Chunk,
    ChunkMeta,
    EvidenceSpan,
    NormalizedRequirement,
    ScoringArtifact,
    SkillVerdict,
)
from rag.config import RagSettings
from rag.hashing import content_id


def make_base_candidate():
    return {
        "name": "Ada Lovelace",
        "skills": ["python", "machine learning", "sql", "docker"],
        "experience_years": 6.0,
        "experience_breakdown": [
            {"role": "ML Engineer", "type": "full-time", "duration_years": 4},
            {"role": "Data Science Intern", "type": "internship", "duration_years": 0.5},
        ],
        "education": ["btech", "ms"],
        "projects": [
            "Built a churn prediction pipeline using scikit-learn",
            "Deployed a model to production with Docker and FastAPI",
        ],
        "semantic_summary": "Experienced ML engineer skilled in building and deploying predictive models.",
    }


def roundtrip(instance):
    dumped = instance.model_dump_json()
    model = type(instance)
    return model.model_validate_json(dumped)


base = make_base_candidate()
candidate_id = content_id(base["semantic_summary"])


def make_chunk_meta():
    return ChunkMeta(
        candidate_id=candidate_id,
        chunk_id=f"{candidate_id[:8]}:00",
        section_type="projects",
        chunker_version="v1",
        embed_model="sentence-transformers/all-mpnet-base-v2",
    )


def make_chunk():
    return Chunk(
        chunk_id=f"{candidate_id[:8]}:00",
        candidate_id=candidate_id,
        raw_text=base["projects"][1],
        char_start=0,
        char_end=len(base["projects"][1]),
        meta=make_chunk_meta(),
    )


def make_requirement():
    return NormalizedRequirement(
        skill_name="docker",
        tier="preferred",
        source="llm_classifier",
    )


def make_evidence_span():
    return EvidenceSpan(
        chunk_id=f"{candidate_id[:8]}:00",
        candidate_id=candidate_id,
        text=base["projects"][1],
        dense_rank=1,
        lexical_rank=2,
        fused_rank=1,
        cross_encoder_score=0.87,
    )


def make_verdict():
    return SkillVerdict(
        skill_name="docker",
        verdict="demonstrated",
        cited_chunk_ids=[f"{candidate_id[:8]}:00"],
        quote="Deployed a model to production with Docker and FastAPI.",
    )


def make_candidate_evidence():
    return CandidateEvidence(
        candidate_id=candidate_id,
        verdicts=[make_verdict()],
        rejected_citations=2,
    )


def make_scoring_artifact():
    return ScoringArtifact(
        candidate_id=candidate_id,
        scoring_version="v1",
        signal_breakdown={
            "required_skills": 0.91,
            "semantic": 0.62,
            "reranker": 0.5,
            "experience": 1.0,
            "education": 1.0,
            "projects": 0.8,
            "evidence": 0.75,
        },
        final_score=0.81,
    )


def test_all_schema_roundtrips():
    instances = [
        make_chunk_meta(),
        make_chunk(),
        make_requirement(),
        make_evidence_span(),
        make_verdict(),
        make_candidate_evidence(),
        make_scoring_artifact(),
    ]
    failures = []
    for instance in instances:
        try:
            assert roundtrip(instance) == instance, f"{type(instance).__name__} mismatch"
        except AssertionError as e:
            failures.append(str(e))
    assert not failures, failures
    print(f"ROUNDTRIP OK: {len(instances)} schema types round-tripped via model_dump_json/model_validate_json")


def test_content_id_deterministic():
    text = base["semantic_summary"]
    assert content_id(text) == content_id(text), "content_id must be deterministic"
    print(f"content_id deterministic OK: {content_id(text)}")


def test_content_id_distinct():
    a = content_id(base["semantic_summary"])
    b = content_id(base["semantic_summary"][:-1] + "X")
    assert a != b, "content_id must differ for different inputs"
    print(f"content_id distinct OK: {a} != {b}")


def test_rag_settings_defaults():
    settings = RagSettings()
    assert settings.rag_enabled is False, "rag_enabled must default to False"
    assert settings.qdrant_url == "http://localhost:6333", "qdrant_url default wrong"
    assert settings.resume_collection_name == "resume_chunks_v1"
    assert settings.esco_collection_name == "skills_esco_v1"
    assert settings.chunker_version == "v1"
    assert settings.embed_model_name == "sentence-transformers/all-mpnet-base-v2"
    assert settings.dense_top_k == 8
    assert settings.lexical_top_k == 8
    assert settings.rerank_top_n == 3
    print("RagSettings defaults OK: rag_enabled=False, qdrant_url='http://localhost:6333', and all default fields confirmed")


if __name__ == "__main__":
    test_all_schema_roundtrips()
    test_content_id_deterministic()
    test_content_id_distinct()
    test_rag_settings_defaults()
    print("ALL TESTS PASSED")
