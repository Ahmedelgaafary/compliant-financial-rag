import pytest

from src.retrieval.evaluation import (
    recall_at_k,
    reciprocal_rank,
)
from src.retrieval.models import RetrievalResult


def _result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="document-1",
        text=f"Text for {chunk_id}",
        score=1.0,
        page_number=1,
        section="Financial Statements",
        document_sha256="a" * 64,
        retrieval_method="hybrid_rrf",
    )


def test_recall_at_one() -> None:
    results = (
        _result("chunk-a"),
        _result("chunk-b"),
        _result("chunk-c"),
    )

    assert recall_at_k(
        results,
        expected_chunk_id="chunk-a",
        k=1,
    ) == 1.0


def test_recall_at_three() -> None:
    results = (
        _result("chunk-a"),
        _result("chunk-b"),
        _result("chunk-c"),
    )

    assert recall_at_k(
        results,
        expected_chunk_id="chunk-c",
        k=3,
    ) == 1.0


def test_recall_returns_zero_when_missing() -> None:
    results = (
        _result("chunk-a"),
        _result("chunk-b"),
    )

    assert recall_at_k(
        results,
        expected_chunk_id="chunk-x",
        k=5,
    ) == 0.0


def test_reciprocal_rank() -> None:
    results = (
        _result("chunk-a"),
        _result("chunk-b"),
        _result("chunk-c"),
    )

    assert reciprocal_rank(
        results,
        expected_chunk_id="chunk-b",
    ) == pytest.approx(0.5)


def test_reciprocal_rank_returns_zero_when_missing() -> None:
    results = (
        _result("chunk-a"),
        _result("chunk-b"),
    )

    assert reciprocal_rank(
        results,
        expected_chunk_id="chunk-x",
    ) == 0.0


def test_recall_rejects_invalid_k() -> None:
    results = (_result("chunk-a"),)

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        recall_at_k(
            results,
            expected_chunk_id="chunk-a",
            k=0,
        )