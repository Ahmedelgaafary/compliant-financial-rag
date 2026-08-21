from src.ingestion.chunker import DocumentChunk
from src.retrieval.vector_store import VectorStore


def _make_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="test-document",
        page_number=1,
        text=text,
        section="Financial Statements",
        document_sha256="a" * 64,
    )


def test_vector_retriever_returns_results() -> None:
    """Vector retriever should return ranked results."""

    chunks = (
        _make_chunk(
            "chunk-1",
            "Revenue was $42.8 billion in fiscal year 2025.",
        ),
        _make_chunk(
            "chunk-2",
            "Cybersecurity risks may affect financial performance.",
        ),
        _make_chunk(
            "chunk-3",
            "The company operates payment services worldwide.",
        ),
    )

    vector_store = VectorStore(chunks)
    results = vector_store.retrieve(
        "What was the company's revenue?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].retrieval_method == "vector"
    assert results[0].chunk_id == "chunk-1"


def test_vector_retriever_preserves_provenance() -> None:
    """Vector retrieval should preserve chunk metadata."""

    chunks = (
        _make_chunk(
            "chunk-1",
            "Revenue was $42.8 billion.",
        ),
    )

    vector_store = VectorStore(chunks)

    results = vector_store.retrieve(
        "revenue",
        top_k=1,
    )

    result = results[0]

    assert result.chunk_id == "chunk-1"
    assert result.document_id == "test-document"
    assert result.page_number == 1
    assert result.section == "Financial Statements"
    assert result.document_sha256 == "a" * 64
    assert result.retrieval_method == "vector"


def test_vector_retriever_rejects_empty_query() -> None:
    """Empty queries should be rejected."""

    chunks = (
        _make_chunk(
            "chunk-1",
            "Revenue was $42.8 billion.",
        ),
    )

    vector_store = VectorStore(chunks)

    try:
        vector_store.retrieve("")
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"
    else:
        raise AssertionError("Expected ValueError")


def test_vector_retriever_rejects_empty_chunks() -> None:
    """Vector retriever should reject an empty corpus."""

    try:
        VectorStore(())
    except ValueError as exc:
        assert str(exc) == "chunks cannot be empty"
    else:
        raise AssertionError("Expected ValueError")