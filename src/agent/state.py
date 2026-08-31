"""State definitions for the compliant financial RAG agent.

The agent state is intentionally backward compatible with the original
single-question workflow while supporting multiple questions and multiple
companies in one request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AuditStatus(StrEnum):
    """Audit lifecycle states."""

    PENDING = "pending"
    IN_REVIEW = "in_review"
    ROUTED = "routed"
    COMPLETED = "completed"


class FinalResponseStatus(StrEnum):
    """Final response lifecycle states."""

    NOT_STARTED = "not_started"
    GENERATED = "generated"
    BLOCKED = "blocked"
    ROUTED_TO_AUDIT = "routed_to_audit"


class HumanDecision(StrEnum):
    """Possible human-review decisions."""

    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"


@dataclass
class QuestionSpec:
    """Normalized representation of one user question.

    A QuestionSpec allows the agent to independently retrieve, extract,
    verify, and answer each question in a larger multi-question request.

    ``company`` is optional because some questions may be company-agnostic.
    """

    question_id: str
    question: str
    company: str | None = None
    metric: str | None = None
    period: str | None = None
    requested_metrics: list[str] = field(default_factory=list)
    requested_periods: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentClaim:
    """Agent-level claim wrapper.

    The underlying deterministic verification layer continues to use
    ``src.verification.models.Claim``.  This class provides a stable
    agent-facing representation for multi-question workflows.
    """

    claim_id: str
    question_id: str | None = None
    company: str | None = None
    subject: str = ""
    value: Any = None
    unit: str | None = None
    period: str | None = None
    source_chunk_id: str | None = None
    source_document_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Complete state carried through the LangGraph workflow.

    The state supports:

    - one question / one company
    - multiple questions
    - multiple companies
    - multiple metrics
    - independent evidence per question
    - independent deterministic verification per question
    - human audit routing

    Backward compatibility is preserved for the original fields used by
    the existing tests and API.
    """

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    user_query: str = ""

    # ------------------------------------------------------------------
    # Query analysis
    # ------------------------------------------------------------------

    query_analysis: dict[str, Any] | None = None

    # Normalized multi-question representation.
    query_tasks: list[dict[str, Any]] = field(default_factory=list)

    # Strongly typed representation available to newer callers.
    question_specs: list[QuestionSpec] = field(default_factory=list)

    # Companies explicitly detected in the request.
    companies: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    # Original aggregate retrieval output.
    retrieval_results: list[Any] = field(default_factory=list)

    # Retrieval grouped by question/task ID.
    retrieval_by_task: dict[str, list[Any]] = field(
        default_factory=dict
    )

    # Retrieval grouped by company.
    retrieval_by_company: dict[str, list[Any]] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Claims
    # ------------------------------------------------------------------

    # Original claim list retained for compatibility.
    claims: list[Any] = field(default_factory=list)

    # Claims grouped by question/task ID.
    claims_by_task: dict[str, list[Any]] = field(default_factory=dict)

    # Claims grouped by company.
    claims_by_company: dict[str, list[Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    verification_results: list[Any] = field(default_factory=list)

    # Verification grouped by question/task ID.
    verification_by_task: dict[str, list[Any]] = field(
        default_factory=dict
    )

    # Verification grouped by company.
    verification_by_company: dict[str, list[Any]] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    # Raw intermediate LLM output.
    raw_llm_output: str = ""

    # Per-question generated answers.
    answers_by_task: dict[str, str] = field(default_factory=dict)

    # Per-company generated answers.
    answers_by_company: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------

    guardrail_result: Any | None = None
    risk_assessment: Any | None = None
    should_route_to_audit: bool = False

    # Per-question guardrail results.
    guardrail_by_task: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    audit_record: Any | None = None
    audit_records: list[Any] = field(default_factory=list)

    audit_status: AuditStatus = AuditStatus.PENDING

    human_decision: HumanDecision | None = None

    # ------------------------------------------------------------------
    # Final response
    # ------------------------------------------------------------------

    final_answer: str = ""

    final_response_status: FinalResponseStatus = (
        FinalResponseStatus.NOT_STARTED
    )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    error: str | None = None

    errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Compatibility / lifecycle methods
    # ------------------------------------------------------------------

    def set_audit_status(self, status: AuditStatus) -> None:
        """Set the audit lifecycle status."""

        if not isinstance(status, AuditStatus):
            status = AuditStatus(status)

        self.audit_status = status

    def set_human_decision(self, decision: HumanDecision) -> None:
        """Set a human decision only after review has started.

        A decision cannot be recorded while the request is still pending.
        """

        if self.audit_status not in {
            AuditStatus.IN_REVIEW,
            AuditStatus.ROUTED,
        }:
            raise ValueError(
                "Human decision cannot be set before the audit enters "
                "review."
            )

        if not isinstance(decision, HumanDecision):
            decision = HumanDecision(decision)

        self.human_decision = decision
        self.audit_status = AuditStatus.COMPLETED

    def set_final_response_status(
        self,
        status: FinalResponseStatus,
    ) -> None:
        """Set the final response lifecycle status."""

        if not isinstance(status, FinalResponseStatus):
            status = FinalResponseStatus(status)

        self.final_response_status = status

    def add_error(self, message: str) -> None:
        """Record an error while preserving the primary error field."""

        message = str(message)

        if not message:
            return

        self.errors.append(message)

        if self.error is None:
            self.error = message