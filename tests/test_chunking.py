from pathlib import Path

import pytest

from src.ingestion.chunker import DocumentChunker
from src.ingestion.pdf_parser import PDFParser

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "sample_financial_report.pdf"
)


def test_chunker_preserves_document_metadata() -> None:
    """Chunks should preserve document-level provenance."""

    parser = PDFParser()
    document = parser.parse(FIXTURE_PATH)

    chunker = DocumentChunker()

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 0

    chunk = chunks[0]

    assert chunk.document_id == document.document_id
    assert chunk.document_sha256 == document.sha256
    assert chunk.page_number == 1
    assert chunk.section == "unknown"


def test_chunker_extracts_text() -> None:
    """Chunks should contain meaningful document text."""

    parser = PDFParser()
    document = parser.parse(FIXTURE_PATH)

    chunker = DocumentChunker()

    chunks = chunker.chunk_document(document)

    combined_text = " ".join(chunk.text for chunk in chunks)

    assert "Example Financial Corp." in combined_text
    assert "$42.8 billion" in combined_text
    assert "$5.2 billion" in combined_text


def test_chunker_generates_deterministic_ids() -> None:
    """The same document should produce the same chunk IDs."""

    parser = PDFParser()
    document = parser.parse(FIXTURE_PATH)

    chunker = DocumentChunker()

    first_chunks = chunker.chunk_document(document)
    second_chunks = chunker.chunk_document(document)

    first_ids = [chunk.chunk_id for chunk in first_chunks]
    second_ids = [chunk.chunk_id for chunk in second_chunks]

    assert first_ids == second_ids


def test_chunker_rejects_invalid_chunk_size() -> None:
    """Chunk size must be positive."""

    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than zero",
    ):
        DocumentChunker(chunk_size=0)


def test_chunker_rejects_invalid_overlap() -> None:
    """Chunk overlap must be smaller than chunk size."""

    with pytest.raises(
        ValueError,
        match="chunk_overlap cannot be negative",
    ):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_chunker_rejects_overlap_equal_to_chunk_size() -> None:
    """Chunk overlap must be smaller than chunk size."""

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=100,
        )


def test_chunker_preserves_paragraph_boundaries() -> None:
    """Chunker should preserve complete paragraphs when they fit."""

    parser = PDFParser()
    document = parser.parse(FIXTURE_PATH)

    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.text.strip() == chunk.text


def test_chunker_splits_long_paragraph() -> None:
    """Long paragraphs should be split into multiple chunks."""

    parser = PDFParser()
    document = parser.parse(FIXTURE_PATH)

    chunker = DocumentChunker(
        chunk_size=50,
        chunk_overlap=10,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.text) <= 50


def test_chunker_supports_zero_overlap() -> None:
    """Chunker should support disabling overlap."""

    parser = PDFParser()
    document = parser.parse(FIXTURE_PATH)

    chunker = DocumentChunker(
        chunk_size=50,
        chunk_overlap=0,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1