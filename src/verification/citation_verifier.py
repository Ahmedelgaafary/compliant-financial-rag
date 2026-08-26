from dataclasses import dataclass

from src.verification.provenance import EvidenceProvenance


@dataclass(frozen=True)
class CitationVerificationResult:
    """Result of deterministic citation provenance validation."""

    valid: bool
    reason: str


class CitationVerifier:
    """Validate evidence citation provenance."""

    def verify(
        self,
        provenance: EvidenceProvenance | None,
        expected_document_id: str | None = None,
        expected_chunk_id: str | None = None,
    ) -> CitationVerificationResult:
        """Validate provenance and optional expected identifiers."""

        if provenance is None:
            return CitationVerificationResult(
                valid=False,
                reason="Citation provenance is missing.",
            )

        if not provenance.is_valid():
            return CitationVerificationResult(
                valid=False,
                reason="Citation provenance is invalid.",
            )

        if (
            expected_document_id is not None
            and provenance.document_id != expected_document_id
        ):
            return CitationVerificationResult(
                valid=False,
                reason=(
                    "Citation document ID does not match "
                    "expected document."
                ),
            )

        if (
            expected_chunk_id is not None
            and provenance.chunk_id != expected_chunk_id
        ):
            return CitationVerificationResult(
                valid=False,
                reason=(
                    "Citation chunk ID does not match "
                    "expected chunk."
                ),
            )

        return CitationVerificationResult(
            valid=True,
            reason="Citation provenance is valid.",
        )