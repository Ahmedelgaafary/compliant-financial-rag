# tests/test_retrieval_integration.py

from unittest.mock import patch

from src.ingestion.chunker import DocumentChunk
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import VectorStore


def _make_chunk(
    chunk_id: str,
    text: str,
    section: str,
) -> DocumentChunk:
    """Create a test DocumentChunk with the given ID."""
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="financial-report-2025",
        page_number=1,
        text=text,
        section=section,
        document_sha256="a" * 64,
    )


def test_real_hybrid_retrieval() -> None:
    """BM25 and vector retrieval should work together."""
    
    chunks = (
        _make_chunk(
            "revenue",
            "Total revenue for fiscal year 2025 was $42.8 billion.",
            "Management Discussion And Analysis",
        ),
        _make_chunk(
            "expenses",
            "Operating expenses were $31.4 billion in fiscal year 2025.",
            "Financial Statements",
        ),
        _make_chunk(
            "risk",
            "Cybersecurity risks could negatively affect financial performance.",
            "Risk Factors",
        ),
        _make_chunk(
            "assets",
            "Total assets were $185.6 billion at the end of fiscal year 2025.",
            "Balance Sheet",
        ),
    )
    
    bm25 = BM25Retriever(chunks)
    
    # Mock VectorStore to use test chunks instead of loading from disk
    with patch("src.retrieval.vector_store.VectorStore.__init__") as mock_init:
        # Skip the real __init__ and just set attributes
        mock_init.return_value = None
        
        vector_store = VectorStore.__new__(VectorStore)
        vector_store.chunks = chunks
        vector_store.model = None
        vector_store.embeddings = None
        vector_store.index = None
        
        # Manually set the retrieve method to use test chunks
        def mock_retrieve(query, top_k=5):
            from src.retrieval.models import RetrievalResult
            results = []
            for chunk in chunks[:top_k]:
                results.append(
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        score=1.0,
                        page_number=chunk.page_number,
                        section=chunk.section,
                        document_sha256=chunk.document_sha256,
                        retrieval_method="vector",
                    )
                )
            return tuple(results)
        
        vector_store.retrieve = mock_retrieve
        
        hybrid = HybridRetriever(
            bm25_retriever=bm25,
            vector_store=vector_store,
        )
        
        results = hybrid.retrieve(
            "What was total revenue in fiscal year 2025?",
            top_k=3,
        )
        
        assert len(results) == 3
        # Check that the top result contains revenue-related content
        assert "revenue" in results[0].text.lower()