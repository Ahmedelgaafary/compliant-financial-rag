from dataclasses import dataclass

from src.verification.audit_queue import AuditQueueItem
from src.verification.audit_queue_manager import AuditQueueManager
from src.verification.models import Claim, VerificationResult
from src.verification.provenance import EvidenceProvenance
from src.verification.risk import (
    RiskAssessment,
    RiskClassifier,
    RiskLevel,
    RoutingDecision,
)


@dataclass(frozen=True)
class VerificationPipelineResult:
    """Complete result of the verification pipeline."""

    claim: Claim
    verification: VerificationResult
    risk: RiskAssessment
    audit_queue_item: AuditQueueItem | None = None


class VerificationPipeline:
    """Coordinate verification, risk classification, and routing."""

    def __init__(
        self,
        claim_verifier,
        risk_classifier: RiskClassifier | None = None,
        audit_queue: AuditQueueManager | None = None,
    ) -> None:
        self._claim_verifier = claim_verifier
        self._risk_classifier = (
            risk_classifier or RiskClassifier()
        )
        self._audit_queue = (
            audit_queue or AuditQueueManager()
        )

    def process(
        self,
        claim: Claim,
        evidence_text: str,
        provenance: EvidenceProvenance,
        queue_id: str | None = None,
    ) -> VerificationPipelineResult:
        """Verify a claim and route it according to risk."""

        verification = self._claim_verifier.verify(
            claim=claim,
            evidence_text=evidence_text,
        )

        risk = self._risk_classifier.classify(
            verification
        )

        audit_queue_item = None

        if (
            risk.level == RiskLevel.HIGH
            and risk.decision == RoutingDecision.HUMAN_AUDIT
        ):
            if queue_id is None:
                raise ValueError(
                    "queue_id is required for human audit routing."
                )

            audit_queue_item = AuditQueueItem.create(
                queue_id=queue_id,
                claim=claim,
                verification=verification,
                risk=risk,
                provenance=provenance,
                evidence_text=evidence_text,
            )

            self._audit_queue.add(audit_queue_item)

        return VerificationPipelineResult(
            claim=claim,
            verification=verification,
            risk=risk,
            audit_queue_item=audit_queue_item,
        )

    @property
    def audit_queue(self) -> AuditQueueManager:
        """Return the audit queue manager."""

        return self._audit_queue