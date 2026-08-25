from dataclasses import dataclass
from enum import StrEnum

from src.verification.models import (
    VerificationResult,
    VerificationStatus,
)


class RiskLevel(StrEnum):
    """Risk levels assigned to verification outcomes."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RoutingDecision(StrEnum):
    """Action selected for a verification result."""

    AUTO_APPROVE = "auto_approve"
    REVIEW = "review"
    HUMAN_AUDIT = "human_audit"


@dataclass(frozen=True)
class RiskAssessment:
    """Immutable risk and routing assessment."""

    level: RiskLevel
    decision: RoutingDecision
    reason: str


class RiskClassifier:
    """Classify verification results and determine routing."""

    def classify(
        self,
        result: VerificationResult,
    ) -> RiskAssessment:
        """Return a deterministic risk assessment."""

        if result.status == VerificationStatus.VERIFIED:
            if result.confidence >= 0.99:
                return RiskAssessment(
                    level=RiskLevel.LOW,
                    decision=RoutingDecision.AUTO_APPROVE,
                    reason="Verified with high confidence.",
                )

            return RiskAssessment(
                level=RiskLevel.MEDIUM,
                decision=RoutingDecision.REVIEW,
                reason="Verified but confidence is below the "
                "automatic approval threshold.",
            )

        if result.status == VerificationStatus.INCONCLUSIVE:
            return RiskAssessment(
                level=RiskLevel.HIGH,
                decision=RoutingDecision.HUMAN_AUDIT,
                reason="Claim could not be deterministically verified.",
            )

        if result.status == VerificationStatus.REJECTED:
            return RiskAssessment(
                level=RiskLevel.HIGH,
                decision=RoutingDecision.HUMAN_AUDIT,
                reason="Evidence contradicts the claim.",
            )

        return RiskAssessment(
            level=RiskLevel.HIGH,
            decision=RoutingDecision.HUMAN_AUDIT,
            reason="Unknown verification status.",
        )