from collections.abc import Sequence

from src.retrieval.models import RetrievalResult


def recall_at_k(
    results: Sequence[RetrievalResult],
    expected_chunk_id: str,
    k: int,
) -> float:
    """Calculate Recall@K for a single query."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    top_results = results[:k]

    return float(
        any(
            result.chunk_id == expected_chunk_id
            for result in top_results
        )
    )


def reciprocal_rank(
    results: Sequence[RetrievalResult],
    expected_chunk_id: str,
) -> float:
    """Calculate reciprocal rank for one query."""

    for rank, result in enumerate(results, start=1):
        if result.chunk_id == expected_chunk_id:
            return 1.0 / rank

    return 0.0