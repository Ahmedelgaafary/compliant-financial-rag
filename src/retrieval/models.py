from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    """A single document retrieved for a user query."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    page_number: int
    section: str
    document_sha256: str
    retrieval_method: str