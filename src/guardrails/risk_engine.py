"""Deterministic risk assessment engine."""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.models import RetrievalResult
from src.verification.models import (
    VerificationResult,
    VerificationStatus,
)

from .confidence import ConfidenceScore
from .policies import GuardrailPolicies


@dataclass
class RiskAssessment:
    """Deterministic risk decision."""

    risk_score: float
    risk_level: str
    triggers: list[str]
    recommended_action: str


class RiskEngine:
    """Convert verification and confidence signals into risk."""

    def __init__(
        self,
        policies: GuardrailPolicies,
    ) -> None:
        self.policies = policies

    def assess(
        self,
        retrieval_results: list[RetrievalResult],
        verification_results: list[VerificationResult],
        confidence: ConfidenceScore,
    ) -> RiskAssessment:
        """Calculate a deterministic risk score."""

        risk_score = 0.0
        triggers: list[str] = []

        rejected = [
            result
            for result in verification_results
            if result.status
            == VerificationStatus.REJECTED
        ]

        if rejected:
            risk_score += (
                self.policies.risk_increment_rejected
                * len(rejected)
            )

            triggers.extend(
                f"REJECTED_{result.reason}"
                for result in rejected
            )

        inconclusive = [
            result
            for result in verification_results
            if result.status
            == VerificationStatus.INCONCLUSIVE
        ]

        if inconclusive:
            risk_score += (
                self.policies.risk_increment_inconclusive
                * len(inconclusive)
            )

            triggers.extend(
                f"INCONCLUSIVE_{result.reason}"
                for result in inconclusive
            )

        contradictions = [
            result
            for result in verification_results
            if result.reason.casefold()
            == "evidence_contradicts"
        ]

        if contradictions:
            risk_score += (
                self.policies.risk_increment_contradiction
                * len(contradictions)
            )
            triggers.append(
                "EVIDENCE_CONTRADICTS"
            )

        if (
            confidence.overall
            < self.policies.min_overall_confidence
        ):
            risk_score += (
                self.policies.risk_increment_low_confidence
            )
            triggers.append(
                "LOW_CONFIDENCE"
            )

        missing_provenance = [
            result
            for result in verification_results
            if result.evidence_chunk_id is None
        ]

        if missing_provenance:
            risk_score += (
                self.policies.risk_increment_missing_provenance
                * len(missing_provenance)
            )
            triggers.append(
                "MISSING_PROVENANCE"
            )

        if not retrieval_results:
            risk_score += (
                self.policies.risk_increment_no_evidence
            )
            triggers.append(
                "NO_EVIDENCE"
            )

        elif len(retrieval_results) < (
            self.policies.min_evidence_chunks
        ):
            risk_score += (
                self.policies.risk_increment_insufficient_evidence
            )
            triggers.append(
                "INSUFFICIENT_EVIDENCE"
            )

        numeric_mismatch = any(
            result.reason.casefold()
            == "numeric_mismatch"
            for result in rejected
        )

        if numeric_mismatch:
            risk_score += (
                self.policies.risk_increment_numeric_mismatch
            )
            triggers.append(
                "NUMERIC_MISMATCH"
            )

        risk_score = min(
            1.0,
            risk_score,
        )

        risk_level = self.policies.get_risk_level(
            risk_score
        )

        if (
            self.policies.block_on_numeric_mismatch
            and numeric_mismatch
        ):
            recommended_action = "BLOCK"

        elif risk_level == "HIGH":
            recommended_action = "HUMAN_REVIEW"

        elif risk_level == "MEDIUM":
            recommended_action = (
                "AUTO_ANSWER_WITH_DISCLAIMER"
            )

        else:
            recommended_action = "AUTO_ANSWER"

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level,
            triggers=triggers,
            recommended_action=recommended_action,
        )