from dataclasses import dataclass
from pathlib import Path

import fitz

from src.exceptions import DocumentProcessingError
from src.ingestion.document_hash import calculate_sha256
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ParsedPage:
    """Text and metadata extracted from a single PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    """Parsed representation of a financial PDF document."""

    document_id: str
    file_name: str
    file_path: str
    page_count: int
    pages: tuple[ParsedPage, ...]
    sha256: str


class PDFParser:
    """Extract page-level text from PDF financial documents."""

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse a PDF and return its extracted page-level text.

        Args:
            file_path: Path to the PDF document.

        Returns:
            ParsedDocument containing extracted pages and document hash.

        Raises:
            DocumentProcessingError:
                If the file is invalid or cannot be parsed.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise DocumentProcessingError(
                f"Document does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise DocumentProcessingError(
                f"Document path is not a file: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise DocumentProcessingError(
                f"Expected a PDF file, received: {file_path.suffix}"
            )

        logger.info("Parsing PDF: %s", file_path.name)

        try:
            document_hash = calculate_sha256(file_path)

            with fitz.open(file_path) as document:
                pages = tuple(
                    ParsedPage(
                        page_number=page_number,
                        text=page.get_text("text").strip(),
                    )
                    for page_number, page in enumerate(document, start=1)
                )

                document_id = self._generate_document_id(file_path)

                parsed_document = ParsedDocument(
                    document_id=document_id,
                    file_name=file_path.name,
                    file_path=str(file_path.resolve()),
                    page_count=len(pages),
                    pages=pages,
                    sha256=document_hash,
                )

        except fitz.FileDataError as exc:
            raise DocumentProcessingError(
                f"Unable to read PDF: {file_path}"
            ) from exc

        except DocumentProcessingError:
            raise

        except Exception as exc:
            raise DocumentProcessingError(
                f"Unexpected error while parsing PDF: {file_path}"
            ) from exc

        logger.info(
            "PDF parsed successfully: %s pages=%d sha256=%s",
            file_path.name,
            parsed_document.page_count,
            parsed_document.sha256,
        )

        return parsed_document

    @staticmethod
    def _generate_document_id(file_path: Path) -> str:
        """Generate a deterministic document identifier from the file path."""

        return file_path.stem