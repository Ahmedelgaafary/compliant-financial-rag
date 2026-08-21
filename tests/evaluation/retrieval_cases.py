from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCase:
    """A labeled retrieval evaluation case."""

    query: str
    expected_chunk_id: str


RETRIEVAL_CASES = (
    RetrievalCase(
        query="What was total revenue in fiscal year 2025?",
        expected_chunk_id="revenue",
    ),
    RetrievalCase(
        query="How much were operating expenses in 2025?",
        expected_chunk_id="expenses",
    ),
    RetrievalCase(
        query="What cybersecurity risk does the company face?",
        expected_chunk_id="risk",
    ),
    RetrievalCase(
        query="What were total assets in fiscal year 2025?",
        expected_chunk_id="assets",
    ),
)