"""
Purpose: Evaluate the overall confidence of the answer based on retrieval scores, verification status, and other signals.
"""

# src/guardrails/confidence.py
from dataclasses import dataclass
from typing import List

from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult


@dataclass
class ConfidenceScore:
    """Aggregated confidence score and breakdown."""
    overall: float          # 0.0–1.0
    retrieval_confidence: float
    verification_confidence: float
    evidence_sufficiency: float
    contradictions_penalty: float


class ConfidenceScorer:
    """
    Computes a confidence score for the final answer.
    - Higher retrieval scores (BM25 + vector) increase confidence.
    - Verified claims increase confidence.
    - Contradictions or missing evidence decrease confidence.
    """

    def __init__(
        self,
        retrieval_weight: float = 0.4,
        verification_weight: float = 0.6,
    ):
        self.retrieval_weight = retrieval_weight
        self.verification_weight = verification_weight

    def compute(
        self,
        retrieval_results: List[RetrievalResult],
        verification_results: List[VerificationResult],
    ) -> ConfidenceScore:
        """
        Compute confidence based on all evidence and verification outcomes.
        """
        # Example logic:
        avg_retrieval_score = sum(r.score for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0.0

        # Verification confidence: ratio of VERIFIED vs total claims
        verified_count = sum(1 for v in verification_results if v.status == "VERIFIED")
        total_claims = len(verification_results) or 1
        verification_conf = verified_count / total_claims

        # Penalty for contradictions
        contradictions = sum(1 for v in verification_results if v.reason == "EVIDENCE_CONTRADICTS")
        penalty = 0.2 * min(contradictions, 2)  # max penalty 0.4

        overall = (
            self.retrieval_weight * avg_retrieval_score +
            self.verification_weight * verification_conf
        ) - penalty

        overall = max(0.0, min(1.0, overall))

        return ConfidenceScore(
            overall=overall,
            retrieval_confidence=avg_retrieval_score,
            verification_confidence=verification_conf,
            evidence_sufficiency=len(retrieval_results) / 5.0 if retrieval_results else 0.0,  # example threshold
            contradictions_penalty=penalty,
        )