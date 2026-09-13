from pydantic_settings import BaseSettings


class RagSettings(BaseSettings):
    rag_enabled: bool = False
    qdrant_url: str = "http://localhost:6333"
    resume_collection_name: str = "resume_chunks_v1"
    esco_collection_name: str = "skills_esco_v1"
    chunker_version: str = "v1"
    embed_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    dense_top_k: int = 8
    lexical_top_k: int = 8
    rerank_top_n: int = 3
    # --- Phase 10: caching, logging, cost/latency guards -------------------
    llm_cache_enabled: bool = True
    llm_cache_dir: str = ".rag_cache/llm"
    llm_log_enabled: bool = True
    llm_log_path: str = ".rag_logs/llm_calls.jsonl"
    llm_max_concurrency: int = 4
    # Per-job cost ceiling (max LLM calls per process-wide job window).
    # When exhausted, LLM call sites fall back to deterministic paths:
    #   extraction → rule_based_fallback_extraction
    #   classify   → keyword_fallback (split_jd_sections)
    #   evidence   → cross-encoder score thresholds (0.6/0.3)
    # Set to 0 to disable the ceiling (unlimited).
    llm_max_calls_per_job: int = 50
