from dataclasses import dataclass
from datetime import datetime, timezone

from src.verification.models import (
    Claim,
    VerificationResult,
    VerificationStatus,
)
from src.verification.provenance import EvidenceProvenance


@dataclass(frozen=True)
class AuditRecord:
    """Immutable audit record for a financial claim."""

    claim: Claim
    verification: VerificationResult
    provenance: EvidenceProvenance
    evidence_text: str
    created_at: datetime

    @classmethod
    def create(
        cls,
        claim: Claim,
        verification: VerificationResult,
        provenance: EvidenceProvenance,
        evidence_text: str,
    ) -> "AuditRecord":
        """Create an audit record with a UTC timestamp."""

        return cls(
            claim=claim,
            verification=verification,
            provenance=provenance,
            evidence_text=evidence_text,
            created_at=datetime.now(timezone.utc),
        )

    def is_auditable(self) -> bool:
        """Return whether the record contains sufficient audit information."""

        return (
            bool(self.evidence_text.strip())
            and self.provenance.is_valid()
            and self.verification.evidence_chunk_id
            == self.provenance.chunk_id
        )

    def is_verified(self) -> bool:
        """Return whether the underlying claim is verified."""

        return (
            self.verification.status
            == VerificationStatus.VERIFIED
        )