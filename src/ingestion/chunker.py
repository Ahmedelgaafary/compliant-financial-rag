from dataclasses import dataclass
from hashlib import sha256

from src.ingestion.pdf_parser import ParsedDocument
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable chunk extracted from a financial document."""

    chunk_id: str
    document_id: str
    page_number: int
    text: str
    section: str
    document_sha256: str


class DocumentChunker:
    """Create searchable chunks from parsed financial documents."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        document: ParsedDocument,
    ) -> tuple[DocumentChunk, ...]:
        """
        Split a parsed document into searchable chunks.

        Args:
            document: Parsed financial document.

        Returns:
            Tuple of document chunks.
        """

        chunks: list[DocumentChunk] = []

        for page in document.pages:
            if not page.text.strip():
                continue

            page_chunks = self._chunk_text(page.text)

            for chunk_index, text in enumerate(page_chunks):
                chunk_id = self._generate_chunk_id(
                    document.document_id,
                    page.page_number,
                    chunk_index,
                    text,
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        page_number=page.page_number,
                        text=text,
                        section="unknown",
                        document_sha256=document.sha256,
                    )
                )

        logger.info(
            "Document chunked: document=%s chunks=%d",
            document.document_id,
            len(chunks),
        )

        return tuple(chunks)

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into paragraph-aware chunks with overlap."""

        paragraphs = [
            " ".join(paragraph.split())
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        if not paragraphs:
            return []

        chunks: list[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                chunks.extend(
                    self._split_long_paragraph(paragraph)
                )
                continue

            candidate = (
                paragraph
                if not current_chunk
                else f"{current_chunk} {paragraph}"
            )

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                chunks.append(current_chunk)

                overlap = self._get_overlap(current_chunk)

                current_chunk = (
                    f"{overlap} {paragraph}".strip()
                )

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_long_paragraph(
        self,
        paragraph: str,
    ) -> list[str]:
        """Split a paragraph that exceeds the chunk size."""

        chunks: list[str] = []

        start = 0
        text_length = len(paragraph)

        while start < text_length:
            end = min(
                start + self.chunk_size,
                text_length,
            )

            chunk = paragraph[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - self.chunk_overlap

        return chunks

    def _get_overlap(self, text: str) -> str:
        """Return the trailing overlap portion of a chunk."""

        if self.chunk_overlap == 0:
            return ""

        return text[-self.chunk_overlap :]

    @staticmethod
    def _generate_chunk_id(
        document_id: str,
        page_number: int,
        chunk_index: int,
        text: str,
    ) -> str:
        """Generate a deterministic identifier for a document chunk."""

        value = (
            f"{document_id}:"
            f"{page_number}:"
            f"{chunk_index}:"
            f"{text}"
        )

        return sha256(value.encode("utf-8")).hexdigest()