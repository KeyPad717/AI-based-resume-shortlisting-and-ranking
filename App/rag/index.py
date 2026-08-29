from rag.config import RagSettings
from rag.schemas import Chunk
from rag.store import VectorStore


class ResumeIndexer:
    """Embeds Chunk.raw_text in batches and upserts to the resume collection."""

    def __init__(self, store, embedding_service):
        self.store = store
        self.embed = embedding_service
        self.settings = RagSettings()

    def index_chunks(self, chunks):
        if not chunks:
            return
        texts = [chunk.raw_text for chunk in chunks]
        vectors = self.embed.encode(texts)
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "candidate_id": chunk.candidate_id,
                "chunk_id": chunk.chunk_id,
                "section_type": chunk.meta.section_type,
                "chunker_version": chunk.meta.chunker_version,
                "embed_model": chunk.meta.embed_model,
                "raw_text": chunk.raw_text,
            }
            points.append(
                {"id": chunk.chunk_id, "vector": vector, "payload": payload}
            )
        self.store.upsert(self.settings.resume_collection_name, points)


class OntologyIndexer:
    """Indexes ESCO skill concepts into a separate collection."""

    def __init__(self, store, embedding_service):
        self.store = store
        self.embed = embedding_service
        self.settings = RagSettings()

    def index_esco(self, concepts):
        if not concepts:
            return
        texts = []
        for concept in concepts:
            preferred = concept.get("preferred_label", "")
            alt_labels = concept.get("alt_labels", []) or []
            texts.append(preferred + " | " + ", ".join(alt_labels))
        vectors = self.embed.encode(texts)

        points = []
        for concept, vector in zip(concepts, vectors):
            preferred = concept.get("preferred_label", "")
            alt_labels = list(concept.get("alt_labels", []) or [])
            concept_uri = concept.get("concept_uri", "")
            point_id = concept_uri or preferred or "concept"
            payload = {
                "preferred_label": preferred,
                "alt_labels": alt_labels,
                "concept_uri": concept_uri,
                "text": preferred + " | " + ", ".join(alt_labels),
            }
            points.append({"id": point_id, "vector": vector, "payload": payload})
        self.store.upsert(self.settings.esco_collection_name, points)
