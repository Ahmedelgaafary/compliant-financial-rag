from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from src.verification.models import (
    Claim,
    VerificationResult,
)
from src.verification.provenance import EvidenceProvenance
from src.verification.risk import (
    RiskAssessment,
    RiskLevel,
    RoutingDecision,
)


class AuditQueueStatus(StrEnum):
    """Lifecycle states for human audit items."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class AuditQueueItem:
    """Immutable item awaiting human audit."""

    queue_id: str
    claim: Claim
    verification: VerificationResult
    risk: RiskAssessment
    provenance: EvidenceProvenance
    evidence_text: str
    status: AuditQueueStatus
    created_at: datetime

    @classmethod
    def create(
        cls,
        queue_id: str,
        claim: Claim,
        verification: VerificationResult,
        risk: RiskAssessment,
        provenance: EvidenceProvenance,
        evidence_text: str,
    ) -> "AuditQueueItem":
        """Create a pending human audit item."""

        if risk.level != RiskLevel.HIGH:
            raise ValueError(
                "Only high-risk claims can enter the human audit queue."
            )

        if risk.decision != RoutingDecision.HUMAN_AUDIT:
            raise ValueError(
                "Audit queue requires a HUMAN_AUDIT routing decision."
            )

        return cls(
            queue_id=queue_id,
            claim=claim,
            verification=verification,
            risk=risk,
            provenance=provenance,
            evidence_text=evidence_text,
            status=AuditQueueStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )

    def approve(self) -> "AuditQueueItem":
        """Return a new queue item marked as approved."""

        return AuditQueueItem(
            queue_id=self.queue_id,
            claim=self.claim,
            verification=self.verification,
            risk=self.risk,
            provenance=self.provenance,
            evidence_text=self.evidence_text,
            status=AuditQueueStatus.APPROVED,
            created_at=self.created_at,
        )

    def reject(self) -> "AuditQueueItem":
        """Return a new queue item marked as rejected."""

        return AuditQueueItem(
            queue_id=self.queue_id,
            claim=self.claim,
            verification=self.verification,
            risk=self.risk,
            provenance=self.provenance,
            evidence_text=self.evidence_text,
            status=AuditQueueStatus.REJECTED,
            created_at=self.created_at,
        )