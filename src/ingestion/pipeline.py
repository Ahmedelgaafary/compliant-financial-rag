from dataclasses import dataclass
from pathlib import Path

from src.ingestion.chunker import DocumentChunk, DocumentChunker
from src.ingestion.pdf_parser import ParsedDocument, PDFParser
from src.ingestion.section_detector import (
    FinancialSectionDetector,
    Section,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Result produced by the document ingestion pipeline."""

    document: ParsedDocument
    sections: tuple[Section, ...]
    chunks: tuple[DocumentChunk, ...]


class FinancialDocumentPipeline:
    """Orchestrate parsing, section detection, and chunking."""

    def __init__(
        self,
        parser: PDFParser | None = None,
        section_detector: FinancialSectionDetector | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.parser = parser or PDFParser()
        self.section_detector = (
            section_detector or FinancialSectionDetector()
        )
        self.chunker = chunker or DocumentChunker()

    def process(self, file_path: Path) -> IngestionResult:
        """
        Process a financial PDF through the ingestion pipeline.

        Args:
            file_path: Path to the financial PDF.

        Returns:
            IngestionResult containing the document,
            detected sections, and searchable chunks.
        """

        logger.info(
            "Starting ingestion pipeline: %s",
            file_path,
        )

        document = self.parser.parse(file_path)

        full_text = "\n\n".join(
            page.text
            for page in document.pages
            if page.text.strip()
        )

        sections = self.section_detector.detect(full_text)

        chunks = self.chunker.chunk_document(document)

        result = IngestionResult(
            document=document,
            sections=sections,
            chunks=chunks,
        )

        logger.info(
            "Ingestion pipeline completed: "
            "document=%s sections=%d chunks=%d",
            document.document_id,
            len(result.sections),
            len(result.chunks),
        )

        return result