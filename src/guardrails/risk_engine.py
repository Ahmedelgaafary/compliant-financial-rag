# src/guardrails/risk_engine.py
from dataclasses import dataclass
from typing import List

from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult, VerificationStatus

from .confidence import ConfidenceScore
from .policies import GuardrailPolicies


@dataclass
class RiskAssessment:
    """Deterministic risk decision."""
    risk_score: float
    risk_level: str
    triggers: List[str]
    recommended_action: str


class RiskEngine:
    """
    Converts verification results and confidence into a deterministic risk decision.
    It does NOT perform verification itself.
    """

    def __init__(self, policies: GuardrailPolicies):
        self.policies = policies

    def assess(
        self,
        retrieval_results: List[RetrievalResult],
        verification_results: List[VerificationResult],
        confidence: ConfidenceScore,
    ) -> RiskAssessment:
        risk_score = 0.0
        triggers = []

        # 1. Verification failures
        rejected = [
            v for v in verification_results
            if v.status == VerificationStatus.REJECTED
        ]
        if rejected:
            risk_score += (
                self.policies.risk_increment_rejected * len(rejected)
            )
            triggers.extend(
                [f"REJECTED_{v.reason}" for v in rejected]
            )

        inconclusive = [
            v for v in verification_results
            if v.status == VerificationStatus.INCONCLUSIVE
        ]
        if inconclusive:
            risk_score += (
                self.policies.risk_increment_inconclusive * len(inconclusive)
            )
            triggers.extend(
                [f"INCONCLUSIVE_{v.reason}" for v in inconclusive]
            )

        # 2. Contradictions
        contradictions = [
            v for v in verification_results
            if v.reason == "evidence_contradicts"
        ]
        if contradictions:
            risk_score += (
                self.policies.risk_increment_contradiction
                * len(contradictions)
            )
            triggers.append("EVIDENCE_CONTRADICTS")

        # 3. Low confidence
        if confidence.overall < self.policies.min_overall_confidence:
            risk_score += self.policies.risk_increment_low_confidence
            triggers.append("LOW_CONFIDENCE")

        # 4. Missing provenance (chunk_id)
        missing_provenance = [
            v for v in verification_results
            if v.evidence_chunk_id is None
        ]
        if missing_provenance:
            risk_score += (
                self.policies.risk_increment_missing_provenance
                * len(missing_provenance)
            )
            triggers.append("MISSING_PROVENANCE")

        # 5. Insufficient evidence
        if not retrieval_results:
            risk_score += self.policies.risk_increment_no_evidence
            triggers.append("NO_EVIDENCE")
        elif len(retrieval_results) < self.policies.min_evidence_chunks:
            risk_score += (
                self.policies.risk_increment_insufficient_evidence
            )
            triggers.append("INSUFFICIENT_EVIDENCE")

        # 6. Numeric mismatch (critical)
        numeric_mismatch = any(
            v.reason == "numeric_mismatch" for v in rejected
        )
        if numeric_mismatch:
            risk_score += self.policies.risk_increment_numeric_mismatch
            triggers.append("NUMERIC_MISMATCH")

        # Cap risk score
        risk_score = min(1.0, risk_score)

        # Determine risk level and action
        risk_level = self.policies.get_risk_level(risk_score)

        # Only block if explicitly configured and numeric mismatch exists
        if self.policies.block_on_numeric_mismatch and numeric_mismatch:
            recommended_action = "BLOCK"
        elif risk_level == "HIGH":
            recommended_action = "HUMAN_REVIEW"
        elif risk_level == "MEDIUM":
            recommended_action = "AUTO_ANSWER_WITH_DISCLAIMER"
        else:
            recommended_action = "AUTO_ANSWER"

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level,
            triggers=triggers,
            recommended_action=recommended_action,
        )