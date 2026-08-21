from src.ingestion.chunker import DocumentChunk
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import VectorStore


def _make_chunk(
    chunk_id: str,
    text: str,
    section: str,
) -> DocumentChunk:
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
            "Total revenue for fiscal year 2025 "
            "was $42.8 billion.",
            "Management Discussion And Analysis",
        ),
        _make_chunk(
            "expenses",
            "Operating expenses were $31.4 billion "
            "in fiscal year 2025.",
            "Financial Statements",
        ),
        _make_chunk(
            "risk",
            "Cybersecurity risks could negatively "
            "affect financial performance.",
            "Risk Factors",
        ),
        _make_chunk(
            "assets",
            "Total assets were $185.6 billion "
            "at the end of fiscal year 2025.",
            "Balance Sheet",
        ),
    )

    bm25 = BM25Retriever(chunks)

    vector_store = VectorStore(chunks)

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_store=vector_store,
    )

    results = hybrid.retrieve(
        "What was total revenue in fiscal year 2025?",
        top_k=3,
    )

    assert len(results) == 3

    assert results[0].chunk_id == "revenue"

    assert results[0].section == (
        "Management Discussion And Analysis"
    )

    assert results[0].document_sha256 == "a" * 64

    assert results[0].retrieval_method == "hybrid_rrf"