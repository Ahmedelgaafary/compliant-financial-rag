"""
Final safety gate for automatic answers.
Validates that all required conditions are met before releasing output.
Does NOT modify existing guardrails, validation, or verification logic.
"""

from dataclasses import dataclass
from typing import List

from src.guardrails.risk_engine import RiskAssessment
from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult


@dataclass
class FinalSafetyResult:
    allowed: bool
    reasons: List[str]


class FinalSafetyValidator:
    """
    Final gate before sending an answer to the user.
    Checks safety conditions without modifying existing guardrails.
    """

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def validate(
        self,
        generated_answer: str,
        verification_results: List[VerificationResult],
        retrieval_results: List[RetrievalResult],
        risk_assessment: RiskAssessment,
        confidence_score: float,
    ) -> FinalSafetyResult:
        reasons = []

        # 1. Verification completed
        if not verification_results:
            reasons.append("NO_VERIFICATION_RESULTS")

        # 2. Evidence exists
        if not retrieval_results:
            reasons.append("NO_EVIDENCE")

        # 3. Provenance exists (document_id, chunk_id, sha256)
        for r in retrieval_results:
            if not r.document_id or not r.chunk_id or not r.document_sha256:
                reasons.append(f"MISSING_PROVENANCE for chunk {r.chunk_id}")
                break

        # 4. Risk policy allows automatic response
        if risk_assessment.recommended_action not in (
            "AUTO_ANSWER",
            "AUTO_ANSWER_WITH_DISCLAIMER",
        ):
            reasons.append(
                f"RISK_POLICY_BLOCKS_AUTO_ANSWER: {risk_assessment.recommended_action}"
            )

        # 5. No unresolved contradiction
        for v in verification_results:
            if v.reason.lower() == "evidence_contradicts":
                reasons.append("UNRESOLVED_CONTRADICTION")
                break

        # 6. Required confidence condition
        if confidence_score < self.min_confidence:
            reasons.append("LOW_CONFIDENCE")

        return FinalSafetyResult(allowed=len(reasons) == 0, reasons=reasons)