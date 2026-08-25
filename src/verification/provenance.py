from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceProvenance:
    """Immutable provenance information for evidence."""

    document_id: str
    document_hash: str
    chunk_id: str
    page_number: int
    section: str | None = None

    def is_valid(self) -> bool:
        """Return whether the provenance contains required fields."""

        if not self.document_id.strip():
            return False

        if not self.document_hash.strip():
            return False

        if not self.chunk_id.strip():
            return False

        if self.page_number < 1:
            return False

        return True