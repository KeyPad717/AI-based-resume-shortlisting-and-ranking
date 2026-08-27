from dataclasses import dataclass, field

from rag.schemas import Chunk
from rag.chunking import ExtractionError, ResumeChunker


@dataclass
class IngestReport:
    candidate_id: str
    chunk_count: int
    warnings: list = field(default_factory=list)
    coverage: float = 0.0


class ResumeIngestor:
    def __init__(self, chunker=None):
        self.chunker = chunker or ResumeChunker()

    def ingest(self, file_path):
        try:
            chunks = self.chunker.chunk(file_path)
        except ExtractionError as exc:
            report = IngestReport(
                candidate_id="",
                chunk_count=0,
                warnings=[str(exc)],
                coverage=0.0,
            )
            return [], report

        full_text = self.chunker.full_text
        total = len(full_text)
        chunked = sum(chunk.char_end - chunk.char_start for chunk in chunks)
        coverage = (chunked / total) if total else 0.0

        warnings = []
        if coverage < 0.95:
            warnings.append(f"coverage {coverage:.3f} is below the 0.95 threshold")

        report = IngestReport(
            candidate_id=self.chunker.candidate_id,
            chunk_count=len(chunks),
            warnings=warnings,
            coverage=coverage,
        )
        return chunks, report
