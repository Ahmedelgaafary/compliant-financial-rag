""""
Routes the query based on risk assessment.

"""
# src/audit/router.py
from enum import Enum
from dataclasses import dataclass
from typing import List

from src.audit.models import AuditRecord
from src.guardrails.risk_engine import RiskAssessment
from src.verification.models import VerificationResult


class RoutingAction(str, Enum):
    AUTO_ANSWER = "AUTO_ANSWER"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    BLOCK = "BLOCK"


@dataclass
class RoutingDecision:
    action: RoutingAction
    reason: str
    should_create_audit_record: bool
    audit_priority: str  # HIGH, MEDIUM, LOW


class AuditRouter:
    """
    Determines whether to auto-answer, send to human review, or block.
    """

    def __init__(self):
        pass

    def route(
        self,
        risk_assessment: RiskAssessment,
        verification_results: List[VerificationResult],
    ) -> RoutingDecision:
        """
        Makes a routing decision based on the risk assessment and verification outcomes.
        """

        # 1. If risk is HIGH or CRITICAL -> Human Review
        if risk_assessment.risk_level in ["HIGH", "CRITICAL"]:
            return RoutingDecision(
                action=RoutingAction.HUMAN_REVIEW,
                reason=f"Risk level is {risk_assessment.risk_level}. Triggers: {risk_assessment.triggers}",
                should_create_audit_record=True,
                audit_priority="HIGH",
            )

        # 2. If any NUMERIC_MISMATCH exists -> Human Review (per policy)
        for trigger in risk_assessment.triggers:
            if "NUMERIC_MISMATCH" in trigger:
                return RoutingDecision(
                    action=RoutingAction.HUMAN_REVIEW,
                    reason="Numeric mismatch detected. Requires human verification.",
                    should_create_audit_record=True,
                    audit_priority="HIGH",
                )

        # 3. If MEDIUM risk -> Auto-answer with disclaimer, but still audit
        if risk_assessment.risk_level == "MEDIUM":
            return RoutingDecision(
                action=RoutingAction.AUTO_ANSWER,
                reason="Medium risk. Auto-answering with disclaimer. Logging for review.",
                should_create_audit_record=True,  # Still log it for monitoring
                audit_priority="MEDIUM",
            )

        # 4. LOW risk -> Auto-answer
        return RoutingDecision(
            action=RoutingAction.AUTO_ANSWER,
            reason="Low risk. Auto-answering.",
            should_create_audit_record=False,  # Only log if needed
            audit_priority="LOW",
        )