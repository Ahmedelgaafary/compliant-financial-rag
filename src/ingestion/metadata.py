from dataclasses import dataclass
from typing import Literal

DocumentType = Literal[
    "10-K",
    "10-Q",
    "8-K",
    "earnings_report",
    "bank_statement",
    "annual_report",
    "unknown",
]


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata describing a financial document."""

    company: str
    document_type: DocumentType
    fiscal_year: int | None
    reporting_period: str | None
    source: str | None
    sha256: str