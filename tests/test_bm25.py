from src.ingestion.chunker import DocumentChunk
from src.retrieval.bm25 import BM25Retriever


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


def test_bm25_retrieves_relevant_chunk() -> None:
    """BM25 should rank the most relevant financial chunk first."""

    chunks = (
        _make_chunk(
            "chunk-1",
            "Revenue was $42.8 billion in fiscal year 2025.",
        ),
        _make_chunk(
            "chunk-2",
            "The company operates in several global markets.",
        ),
        _make_chunk(
            "chunk-3",
            "Cybersecurity risk may affect financial performance.",
        ),
    )

    retriever = BM25Retriever(chunks)

    results = retriever.retrieve(
        "What was revenue in fiscal year 2025?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"
    assert results[0].retrieval_method == "bm25"


def test_bm25_preserves_provenance() -> None:
    """Retrieval results should preserve chunk provenance."""

    chunks = (
        _make_chunk(
            "chunk-1",
            "Revenue was $42.8 billion.",
        ),
    )

    retriever = BM25Retriever(chunks)

    results = retriever.retrieve(
        "revenue",
        top_k=1,
    )

    result = results[0]

    assert result.chunk_id == "chunk-1"
    assert result.document_id == "test-document"
    assert result.page_number == 1
    assert result.section == "Financial Statements"
    assert result.document_sha256 == "a" * 64


def test_bm25_rejects_empty_query() -> None:
    """Empty queries should be rejected."""

    chunks = (
        _make_chunk(
            "chunk-1",
            "Revenue was $42.8 billion.",
        ),
    )

    retriever = BM25Retriever(chunks)

    try:
        retriever.retrieve("")
    except ValueError as exc:
        assert str(exc) == "query cannot be empty"
    else:
        raise AssertionError("Expected ValueError")


def test_bm25_rejects_empty_chunks() -> None:
    """Retriever should reject an empty corpus."""

    try:
        BM25Retriever(())
    except ValueError as exc:
        assert str(exc) == "chunks cannot be empty"
    else:
        raise AssertionError("Expected ValueError")