# tests/test_vector_store.py


from src.ingestion.chunker import DocumentChunk
from src.retrieval.vector_store import VectorStore


def _make_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:
    """Create a test DocumentChunk with the given ID."""
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
    
    # Force rebuild to use test chunks
    vector_store = VectorStore(chunks, force_rebuild=True)
    results = vector_store.retrieve(
        "What was the company's revenue?",
        top_k=2,
    )
    
    # Check that we got results (could be 1 or 2 depending on model)
    assert len(results) >= 1
    assert results[0].retrieval_method == "vector"
    # Check content instead of ID
    assert "revenue" in results[0].text.lower()


def test_vector_retriever_preserves_provenance() -> None:
    """Vector retrieval should preserve chunk metadata."""
    
    chunks = (
        _make_chunk(
            "chunk-1",
            "Revenue was $42.8 billion.",
        ),
    )
    
    # Force rebuild to use test chunks
    vector_store = VectorStore(chunks, force_rebuild=True)
    results = vector_store.retrieve("revenue", top_k=1)
    result = results[0]
    
    # Check provenance fields are preserved
    assert result.document_id == "test-document"
    assert result.page_number == 1
    assert result.section == "Financial Statements"
    assert result.document_sha256 == "a" * 64
    assert result.retrieval_method == "vector"


def test_vector_retriever_handles_empty_chunks() -> None:
    """Vector retriever should handle empty chunks gracefully."""
    
    # Use force_rebuild=True and index_dir to isolate test
    import shutil
    import tempfile
    
    temp_dir = tempfile.mkdtemp()
    try:
        vector_store = VectorStore(
            chunks=(),
            index_dir=temp_dir,
            force_rebuild=True,
        )
        
        # It should have no chunks and no index
        assert vector_store.chunks == ()
        assert vector_store.index is None
        
        # Retrieval should return empty results
        results = vector_store.retrieve("test query", top_k=5)
        assert results == ()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)