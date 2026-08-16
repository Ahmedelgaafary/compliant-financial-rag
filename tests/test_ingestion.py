from pathlib import Path

import pytest

from src.exceptions import DocumentProcessingError
from src.ingestion.document_hash import calculate_sha256
from src.ingestion.pdf_parser import PDFParser

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "sample_financial_report.pdf"
)


def test_parser_rejects_missing_file() -> None:
    """Parser should reject a file that does not exist."""

    parser = PDFParser()

    with pytest.raises(
        DocumentProcessingError,
        match="Document does not exist",
    ):
        parser.parse(Path("does_not_exist.pdf"))


def test_parser_extracts_pdf_text() -> None:
    """Parser should extract text and page metadata from a PDF."""

    parser = PDFParser()

    document = parser.parse(FIXTURE_PATH)

    assert document.file_name == "sample_financial_report.pdf"
    assert document.page_count == 1
    assert document.document_id == "sample_financial_report"

    assert len(document.pages) == 1

    page = document.pages[0]

    assert page.page_number == 1
    assert "Example Financial Corp." in page.text
    assert "$42.8 billion" in page.text
    assert "$5.2 billion" in page.text

    assert len(document.sha256) == 64
    assert document.sha256 == calculate_sha256(FIXTURE_PATH)


def test_calculate_sha256() -> None:
    """Hash calculation should be deterministic."""

    first_hash = calculate_sha256(FIXTURE_PATH)
    second_hash = calculate_sha256(FIXTURE_PATH)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_calculate_sha256_rejects_missing_file() -> None:
    """Hash calculation should reject missing files."""

    with pytest.raises(
        DocumentProcessingError,
        match="Cannot hash missing file",
    ):
        calculate_sha256(Path("missing_document.pdf"))