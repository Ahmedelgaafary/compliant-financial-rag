"""
Routes queries based on risk assessment and verification outcomes.

The router is deliberately fail-closed:
if the risk assessment is unavailable, the case is sent to
human review rather than attempting to auto-answer.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from src.guardrails.risk_engine import RiskAssessment
from src.verification.models import VerificationResult, VerificationStatus


class RoutingAction(str, Enum):
    AUTO_ANSWER = "AUTO_ANSWER"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    BLOCK = "BLOCK"


@dataclass
class RoutingDecision:
    """Deterministic routing decision for an audit case."""

    action: RoutingAction
    reason: str
    should_create_audit_record: bool
    audit_priority: str  # HIGH, MEDIUM, LOW


class AuditRouter:
    """
    Determines whether a case should be:

    - automatically answered,
    - sent to human review, or
    - blocked.

    The router is fail-closed. Missing risk information is never treated
    as LOW risk because that could incorrectly permit an unsafe answer.
    """

    def __init__(self):
        pass

    def route(
        self,
        risk_assessment: Optional[RiskAssessment],
        verification_results: List[VerificationResult],
    ) -> RoutingDecision:
        """
        Make a deterministic routing decision.

        If the risk assessment is unavailable, conservatively route the
        case to human review with HIGH priority.
        """

        # ---------------------------------------------------------
        # 0. Fail closed when risk assessment is unavailable
        # ---------------------------------------------------------
        if risk_assessment is None:
            return RoutingDecision(
                action=RoutingAction.HUMAN_REVIEW,
                reason=(
                    "Risk assessment unavailable. "
                    "Case conservatively routed to human review."
                ),
                should_create_audit_record=True,
                audit_priority="HIGH",
            )

        # ---------------------------------------------------------
        # 1. HIGH / CRITICAL risk -> Human Review
        # ---------------------------------------------------------
        if risk_assessment.risk_level in {"HIGH", "CRITICAL"}:
            return RoutingDecision(
                action=RoutingAction.HUMAN_REVIEW,
                reason=(
                    f"Risk level is {risk_assessment.risk_level}. "
                    f"Triggers: {risk_assessment.triggers}"
                ),
                should_create_audit_record=True,
                audit_priority="HIGH",
            )

        # ---------------------------------------------------------
        # 2. Verification outcomes
        # ---------------------------------------------------------
        # VerificationStatus is a StrEnum with lowercase values
        # ("verified", "rejected", "inconclusive"). Comparing against
        # hardcoded uppercase string literals here never matched the
        # enum, so INCONCLUSIVE/REJECTED claims silently fell through
        # to the risk-level-only checks below. Compare against the
        # actual enum members instead.
        for verification in verification_results:
            if verification.status == VerificationStatus.INCONCLUSIVE:
                return RoutingDecision(
                    action=RoutingAction.HUMAN_REVIEW,
                    reason=(
                        "Verification inconclusive. "
                        f"Reason: {verification.reason}"
                    ),
                    should_create_audit_record=True,
                    audit_priority="MEDIUM",
                )

            if verification.status == VerificationStatus.REJECTED:
                return RoutingDecision(
                    action=RoutingAction.HUMAN_REVIEW,
                    reason=(
                        "Verification rejected. "
                        f"Reason: {verification.reason}"
                    ),
                    should_create_audit_record=True,
                    audit_priority="HIGH",
                )

        # ---------------------------------------------------------
        # 3. Numeric mismatch -> Human Review
        # ---------------------------------------------------------
        for trigger in risk_assessment.triggers:
            if "NUMERIC_MISMATCH" in trigger:
                return RoutingDecision(
                    action=RoutingAction.HUMAN_REVIEW,
                    reason=(
                        "Numeric mismatch detected. "
                        "Requires human verification."
                    ),
                    should_create_audit_record=True,
                    audit_priority="HIGH",
                )

        # ---------------------------------------------------------
        # 4. MEDIUM risk -> Auto-answer with disclaimer
        # ---------------------------------------------------------
        if risk_assessment.risk_level == "MEDIUM":
            return RoutingDecision(
                action=RoutingAction.AUTO_ANSWER,
                reason=(
                    "Medium risk. Auto-answering with disclaimer. "
                    "Logging for review."
                ),
                should_create_audit_record=True,
                audit_priority="MEDIUM",
            )

        # ---------------------------------------------------------
        # 5. LOW risk -> Auto-answer
        # ---------------------------------------------------------
        return RoutingDecision(
            action=RoutingAction.AUTO_ANSWER,
            reason="Low risk. Auto-answering.",
            should_create_audit_record=False,
            audit_priority="LOW",
        )