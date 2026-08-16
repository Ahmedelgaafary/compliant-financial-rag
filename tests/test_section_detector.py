from pathlib import Path

from src.ingestion.section_detector import FinancialSectionDetector

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "financial_report.txt"
)


def test_detects_financial_sections() -> None:
    """Detector should identify known financial sections."""

    text = FIXTURE_PATH.read_text(encoding="utf-8")

    detector = FinancialSectionDetector()

    sections = detector.detect(text)

    titles = [section.title for section in sections]

    assert "Business Overview" in titles
    assert "Risk Factors" in titles
    assert "Management Discussion And Analysis" in titles
    assert "Financial Statements" in titles
    assert "Notes To The Financial Statements" in titles


def test_section_text_belongs_to_correct_section() -> None:
    """Section content should remain associated with its heading."""

    text = FIXTURE_PATH.read_text(encoding="utf-8")

    detector = FinancialSectionDetector()

    sections = detector.detect(text)

    risk_section = next(
        section
        for section in sections
        if section.title == "Risk Factors"
    )

    assert "regulated financial environment" in risk_section.text
    assert "cybersecurity" in risk_section.text


def test_unknown_headings_are_ignored() -> None:
    """Unknown headings should not create false sections."""

    detector = FinancialSectionDetector()

    text = """
    UNKNOWN SECTION

    Some random content.

    RISK FACTORS

    Regulatory risk exists.
    """

    sections = detector.detect(text)

    titles = [section.title for section in sections]

    assert "Risk Factors" in titles
    assert "Unknown Section" not in titles