"""
Purpose: Compute a risk score for the entire query–answer pipeline, 
considering retrieval confidence, verification outcome, contradictions, etc. 
This is used to decide whether the answer can be returned directly or must go to human audit.
"""
# src/guardrails/risk_engine.py
from dataclasses import dataclass
from typing import List

from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult

from .confidence import ConfidenceScore
from .policies import GuardrailPolicies


@dataclass
class RiskAssessment:
    risk_score: float          # 0.0–1.0
    risk_level: str            # LOW, MEDIUM, HIGH, CRITICAL
    triggers: List[str]        # list of reasons for the risk level
    recommended_action: str    # "AUTO_ANSWER", "HUMAN_REVIEW", "BLOCK"


class RiskEngine:
    """
    Evaluates risk based on all available signals and policies.
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

        # Factor 1: Low confidence
        if confidence.overall < self.policies.min_overall_confidence:
            risk_score += 0.3
            triggers.append("LOW_CONFIDENCE")

        # Factor 2: Verification failures
        failed = [v for v in verification_results if v.status != "VERIFIED"]
        if failed:
            risk_score += 0.2 * len(failed)
            triggers.extend([f"VERIFICATION_FAILURE_{v.reason}" for v in failed])

        # Factor 3: Numeric mismatches (critical)
        numeric_mismatches = [v for v in verification_results if v.reason == "NUMERIC_MISMATCH"]
        if numeric_mismatches:
            risk_score += 0.4
            triggers.append("NUMERIC_MISMATCH")

        # Factor 4: Contradictions
        contradictions = [v for v in verification_results if v.reason == "EVIDENCE_CONTRADICTS"]
        if contradictions:
            risk_score += 0.2 * len(contradictions)
            triggers.append("EVIDENCE_CONTRADICTS")

        # Factor 5: Insufficient evidence
        if not retrieval_results or len(retrieval_results) < 2:
            risk_score += 0.2
            triggers.append("INSUFFICIENT_EVIDENCE")

        # Cap risk score
        risk_score = min(1.0, risk_score)

        # Determine level and action
        risk_level = self.policies.get_risk_level(risk_score)
        if risk_level == "HIGH" or risk_score >= 0.8:
            recommended_action = "HUMAN_REVIEW"
        elif risk_level == "MEDIUM":
            recommended_action = "AUTO_ANSWER_WITH_DISCLAIMER"
        else:
            recommended_action = "AUTO_ANSWER"

        # If critical numeric mismatch and policy says block, override
        if self.policies.block_on_numeric_mismatch and "NUMERIC_MISMATCH" in triggers:
            recommended_action = "BLOCK"

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level,
            triggers=triggers,
            recommended_action=recommended_action,
        )