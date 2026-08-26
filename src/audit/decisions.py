""""
This module evaluates the audit record and suggests a recommended review decision. 
It can be extended with more sophisticated rules (e.g., ML‑based scoring).
"""
# src/audit/decisions.py
from dataclasses import dataclass
from enum import Enum

from src.audit.models import AuditRecord


class ReviewRecommendation(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    NEEDS_MORE_INFO = "NEEDS_MORE_INFO"
    ESCALATE = "ESCALATE"


@dataclass
class DecisionResult:
    recommendation: ReviewRecommendation
    confidence: float  # 0.0–1.0
    reasoning: str
    suggested_notes: str


class DecisionEngine:
    """
    Analyzes an audit record and recommends a review decision.
    This could be rule‑based or use a simple classifier.
    """

    def __init__(self):
        # In real implementation, you might load a model or rules config
        pass

    def analyze(self, record: AuditRecord) -> DecisionResult:
        """
        Based on the record's verification status, risk, and triggers,
        suggest how the human reviewer should decide.
        """
        # Default: approve if everything looks good
       # reasoning = []
        #suggested_notes = ""

        # Rule 1: If verification was VERIFIED and risk is low -> Approve
        if record.verification_status == "VERIFIED" and record.risk_level == "LOW":
            return DecisionResult(
                recommendation=ReviewRecommendation.APPROVE,
                confidence=0.95,
                reasoning="All claims verified. Low risk.",
                suggested_notes="Approve without changes.",
            )

        # Rule 2: If there is a numeric mismatch -> Escalate or Reject
        if "NUMERIC_MISMATCH" in record.triggers:
            return DecisionResult(
                recommendation=ReviewRecommendation.ESCALATE,
                confidence=0.9,
                reasoning="Numeric mismatch requires senior review.",
                suggested_notes="Verify numbers against original documents.",
            )

        # Rule 3: If evidence is missing -> Needs more info
        if record.verification_reason == "EVIDENCE_MISSING":
            return DecisionResult(
                recommendation=ReviewRecommendation.NEEDS_MORE_INFO,
                confidence=0.8,
                reasoning="Insufficient evidence; request additional context.",
                suggested_notes="Please provide missing document pages.",
            )

        # Rule 4: If multiple contradictions -> Reject or Escalate
        if "EVIDENCE_CONTRADICTS" in record.triggers:
            return DecisionResult(
                recommendation=ReviewRecommendation.REJECT,
                confidence=0.7,
                reasoning="Contradictory evidence found; cannot trust the claim.",
                suggested_notes="Reject; ask for clarification.",
            )

        # Default fallback: Need more info
        return DecisionResult(
            recommendation=ReviewRecommendation.NEEDS_MORE_INFO,
            confidence=0.5,
            reasoning="Ambiguous case; manual review required.",
            suggested_notes="Review all evidence and claim consistency.",
        )