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
