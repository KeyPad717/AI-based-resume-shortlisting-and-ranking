from sentence_transformers import SentenceTransformer

from rag.config import RagSettings


class EmbeddingService:
    """Lazy singleton wrapper around SentenceTransformer.

    The model is NOT loaded at import time (that is the existing pipeline.py
    behavior, which this phase explicitly avoids). Loading happens on first
    encode()/warmup()/get_dimension() call.
    """

    _instance = None

    def __init__(self, model_name=None):
        settings = RagSettings()
        self.model_name = model_name or settings.embed_model_name
        self._model = None
        self._batch_size = 32

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def warmup(self):
        """Force model load (for an eventual /warmup endpoint in Phase 7)."""
        self._load()

    def _load(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts):
        # BATCHED: the whole list goes through one model.encode() call, which
        # internally batches — never encode one text at a time. This same
        # batching approach is what should later replace the per-skill
        # encode() calls inside App/pipeline.py's semantic_skill_coverage
        # function (this phase does NOT modify pipeline.py, leaving only this
        # note for the phase that does).
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def get_dimension(self):
        model = self._load()
        return int(model.get_embedding_dimension())
