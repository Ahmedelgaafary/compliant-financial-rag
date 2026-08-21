from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import RetrievalResult


def _result(
    chunk_id: str,
    score: float,
    method: str,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="document-1",
        text=f"Text for {chunk_id}",
        score=score,
        page_number=1,
        section="Financial Statements",
        document_sha256="a" * 64,
        retrieval_method=method,
    )


class FakeBM25:
    """Deterministic BM25 test double."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        return (
            _result("chunk-a", 10.0, "bm25"),
            _result("chunk-b", 8.0, "bm25"),
            _result("chunk-c", 6.0, "bm25"),
        )[:top_k]


class FakeVectorStore:
    """Deterministic vector-store test double."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, ...]:
        return (
            _result("chunk-b", 0.95, "vector"),
            _result("chunk-c", 0.90, "vector"),
            _result("chunk-d", 0.85, "vector"),
        )[:top_k]


def test_hybrid_combines_rankings() -> None:
    """RRF should reward documents appearing in both lists."""

    retriever = HybridRetriever(
        bm25_retriever=FakeBM25(),
        vector_store=FakeVectorStore(),
    )

    results = retriever.retrieve(
        "revenue",
        top_k=4,
    )

    assert len(results) == 4

    chunk_ids = [result.chunk_id for result in results]

    assert chunk_ids[0] == "chunk-b"
    assert chunk_ids[1] == "chunk-c"


def test_hybrid_marks_results_correctly() -> None:
    """Hybrid results should identify their retrieval method."""

    retriever = HybridRetriever(
        bm25_retriever=FakeBM25(),
        vector_store=FakeVectorStore(),
    )

    results = retriever.retrieve(
        "revenue",
        top_k=3,
    )

    assert all(
        result.retrieval_method == "hybrid_rrf"
        for result in results
    )


def test_hybrid_preserves_provenance() -> None:
    """Hybrid retrieval must preserve document provenance."""

    retriever = HybridRetriever(
        bm25_retriever=FakeBM25(),
        vector_store=FakeVectorStore(),
    )

    results = retriever.retrieve(
        "revenue",
        top_k=1,
    )

    result = results[0]

    assert result.document_id == "document-1"
    assert result.page_number == 1
    assert result.section == "Financial Statements"
    assert result.document_sha256 == "a" * 64


def test_hybrid_rejects_empty_query() -> None:
    """Empty queries should be rejected."""

    retriever = HybridRetriever(
        bm25_retriever=FakeBM25(),
        vector_store=FakeVectorStore(),
    )

    try:
        retriever.retrieve("")
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"
    else:
        raise AssertionError("Expected ValueError")


def test_hybrid_rejects_invalid_top_k() -> None:
    """top_k must be positive."""

    retriever = HybridRetriever(
        bm25_retriever=FakeBM25(),
        vector_store=FakeVectorStore(),
    )

    try:
        retriever.retrieve("revenue", top_k=0)
    except ValueError as exc:
        assert str(exc) == "top_k must be greater than zero"
    else:
        raise AssertionError("Expected ValueError")


def test_hybrid_rejects_invalid_rrf_k() -> None:
    """RRF k must be positive."""

    try:
        HybridRetriever(
            bm25_retriever=FakeBM25(),
            vector_store=FakeVectorStore(),
            rrf_k=0,
        )
    except ValueError as exc:
        assert str(exc) == "rrf_k must be greater than zero"
    else:
        raise AssertionError("Expected ValueError")