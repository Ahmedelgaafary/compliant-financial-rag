from pathlib import Path

from src.ingestion.pipeline import FinancialDocumentPipeline

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "sample_financial_report.pdf"
)


def test_financial_document_pipeline() -> None:
    """Pipeline should produce a complete ingestion result."""

    pipeline = FinancialDocumentPipeline()

    result = pipeline.process(FIXTURE_PATH)

    assert result.document.document_id == "sample_financial_report"
    assert result.document.page_count == 1
    assert len(result.document.sha256) == 64

    assert isinstance(result.sections, tuple)
    assert isinstance(result.chunks, tuple)

    assert len(result.chunks) > 0

    chunk = result.chunks[0]

    assert chunk.document_id == result.document.document_id
    assert chunk.document_sha256 == result.document.sha256