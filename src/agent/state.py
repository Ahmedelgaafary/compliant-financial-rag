# src/agent/state.py
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.audit.models import AuditRecord
from src.guardrails.risk_engine import RiskAssessment
from src.guardrails.runner import GuardrailPipelineResult
from src.retrieval.models import RetrievalResult
from src.verification.models import Claim, VerificationResult


class AuditStatus(str, Enum):
    """Track the lifecycle of an audit review."""
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    REJECTED = "REJECTED"


class HumanDecision(str, Enum):
    """Decision made by a human reviewer."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_MORE_INFO = "NEEDS_MORE_INFO"
    ESCALATE = "ESCALATE"


class FinalResponseStatus(str, Enum):
    """Status of the final response to the user."""
    NOT_STARTED = "NOT_STARTED"
    GENERATED = "GENERATED"
    ROUTED_TO_AUDIT = "ROUTED_TO_AUDIT"
    BLOCKED = "BLOCKED"


@dataclass
class AgentState:
    """State for the LangGraph workflow."""

    # -------------------- Input --------------------
    user_query: str = ""

    # -------------------- Query Analysis --------------------
    query_analysis: Optional[dict] = None

    # -------------------- Evidence / Retrieval --------------------
    retrieval_results: List[RetrievalResult] = field(default_factory=list)

    # -------------------- Claims & Verification --------------------
    raw_llm_output: Optional[str] = None
    claims: List[Claim] = field(default_factory=list)
    verification_results: List[VerificationResult] = field(default_factory=list)

    # -------------------- Guardrails & Risk --------------------
    guardrail_result: Optional[GuardrailPipelineResult] = None
    risk_assessment: Optional[RiskAssessment] = None
    should_route_to_audit: bool = False

    # -------------------- Audit & Human Decision --------------------
    audit_record: Optional[AuditRecord] = None
    audit_status: AuditStatus = AuditStatus.PENDING  # new
    human_decision: Optional[HumanDecision] = None   # new

    # -------------------- Final Response --------------------
    final_answer: str = ""
    final_response_status: FinalResponseStatus = FinalResponseStatus.NOT_STARTED  # new
    error: Optional[str] = None

    # -------------------- Convenience methods --------------------
    def set_audit_status(self, status: AuditStatus) -> None:
        """Safely update audit status."""
        self.audit_status = status

    def set_human_decision(self, decision: HumanDecision) -> None:
        """Store the human reviewer's decision."""
        if self.audit_status in (AuditStatus.IN_REVIEW, AuditStatus.RESOLVED):
            self.human_decision = decision
        else:
            raise ValueError(
                "Cannot set human decision unless audit is in review or resolved."
            )

    def set_final_response_status(self, status: FinalResponseStatus) -> None:
        """Update final response status."""
        self.final_response_status = status