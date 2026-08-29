
"""
Main entry point for the human-in-the-loop audit process.

The review service coordinates:
1. Risk/verification routing.
2. Audit-record creation.
3. Reviewer recommendations.
4. Audit logging.
5. Human-review state transitions.

The service is deliberately conservative:
if risk assessment is unavailable, the case is still routed to
human review rather than being silently auto-answered.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from src.audit.audit_log import AuditLogger
from src.audit.decisions import DecisionEngine, ReviewRecommendation
from src.audit.models import AuditRecord, ReviewDecision
from src.audit.queue import AuditQueue
from src.audit.router import AuditRouter, RoutingAction, RoutingDecision
from src.guardrails.risk_engine import RiskAssessment
from src.verification.models import VerificationResult


@dataclass
class ReviewOutcome:
    """Result of the audit process."""

    audit_id: str
    routing_decision: RoutingDecision
    review_recommendation: Optional[ReviewRecommendation]
    final_action: str  # "AUTO_ANSWER", "HUMAN_REVIEW", "BLOCK"
    audit_record: AuditRecord


class ReviewService:
    """
    Orchestrate the human-in-the-loop audit workflow.

    The workflow is:

        risk + verification
                ↓
            AuditRouter
                ↓
        ┌───────┼────────┐
        ↓       ↓        ↓
      AUTO    REVIEW    BLOCK
                ↓
          AuditRecord
                ↓
          AuditQueue
                ↓
        DecisionEngine
    """

    def __init__(
        self,
        router: AuditRouter | None = None,
        queue: AuditQueue | None = None,
        decision_engine: DecisionEngine | None = None,
        logger: AuditLogger | None = None,
    ) -> None:
        self.router = router or AuditRouter()
        self.queue = queue or AuditQueue()
        self.decision_engine = decision_engine or DecisionEngine()
        self.logger = logger or AuditLogger()

    def initiate_review(
        self,
        user_query: str,
        claim: str,
        verification_status: str,
        verification_reason: str,
        risk_assessment: RiskAssessment | None,
        verification_results: List[VerificationResult],
        evidence: List[dict],
        document_id: str,
        document_sha256: str,
        page_number: int,
        provenance: Optional[List[dict]] = None,
    ) -> ReviewOutcome:
        """
        Start the audit process for a case.

        If risk_assessment is unavailable, the router handles the
        condition conservatively and routes the case to human review.

        We do not fabricate a RiskAssessment because doing so would
        undermine the deterministic audit trail.
        """

        # ---------------------------------------------------------
        # 1. Route the case
        # ---------------------------------------------------------
        routing_decision = self.router.route(
            risk_assessment=risk_assessment,
            verification_results=verification_results,
        )

        # ---------------------------------------------------------
        # 2. Create an audit record when required
        # ---------------------------------------------------------
        audit_record: AuditRecord | None = None

        if routing_decision.should_create_audit_record:

            # Build provenance from evidence if it was not supplied.
            if provenance is None:
                provenance = [
                    {
                        "document_id": document_id,
                        "document_sha256": document_sha256,
                        "page_number": item.get(
                            "page",
                            page_number,
                        ),
                        "text": item.get(
                            "text",
                            "",
                        ),
                    }
                    for item in evidence
                ]

            claim_id = ""

            if verification_results:
                claim_id = (
                    getattr(
                        verification_results[0],
                        "claim_id",
                        "",
                    )
                    or ""
                )

            # -----------------------------------------------------
            # Risk fields
            # -----------------------------------------------------
            #
            # When risk assessment is unavailable we preserve the
            # fact that it was unavailable instead of inventing
            # numerical values.
            #
            if risk_assessment is not None:
                risk_level = risk_assessment.risk_level
                risk_assessment_text = str(risk_assessment)
                confidence_score = risk_assessment.risk_score
                risk_score = risk_assessment.risk_score
                triggers = list(risk_assessment.triggers)
            else:
                risk_level = "HIGH"
                risk_assessment_text = (
                    "Risk assessment unavailable. "
                    "Case conservatively routed to human review."
                )
                confidence_score = 0.0
                risk_score = 1.0
                triggers = ["RISK_ASSESSMENT_UNAVAILABLE"]

            # -----------------------------------------------------
            # Create domain audit record
            # -----------------------------------------------------
            audit_record = AuditRecord(
                audit_id="",
                timestamp=datetime.now(),
                user_query=user_query,
                claim=claim,
                verification_status=verification_status,
                verification_reason=verification_reason,
                risk_level=risk_level,
                evidence=evidence,
                provenance=provenance,
                claim_id=claim_id,
                document_id=document_id,
                document_sha256=document_sha256,
                page_number=page_number,
                risk_assessment=risk_assessment_text,
                created_at=datetime.now(),
                confidence_score=confidence_score,
                risk_score=risk_score,
                triggers=triggers,
                verification_results=[
                    v.__dict__
                    for v in verification_results
                ],
            )

            # -----------------------------------------------------
            # Enqueue the audit record
            # -----------------------------------------------------
            audit_id = self.queue.enqueue(audit_record)

            audit_record.audit_id = audit_id

        # ---------------------------------------------------------
        # 3. Generate reviewer recommendation
        # ---------------------------------------------------------
        recommendation = None

        if (
            routing_decision.action == RoutingAction.HUMAN_REVIEW
            and audit_record is not None
        ):
            decision_result = self.decision_engine.analyze(
                audit_record
            )

            recommendation = decision_result.recommendation

            audit_record.review_notes = (
                decision_result.suggested_notes
            )

        # ---------------------------------------------------------
        # 4. Write audit log
        # ---------------------------------------------------------
        if audit_record is not None:
            self.logger.log(audit_record)

        # ---------------------------------------------------------
        # 5. Determine final system action
        # ---------------------------------------------------------
        if routing_decision.action == RoutingAction.HUMAN_REVIEW:
            final_action = "HUMAN_REVIEW"

        elif routing_decision.action == RoutingAction.BLOCK:
            final_action = "BLOCK"

        else:
            final_action = "AUTO_ANSWER"

        return ReviewOutcome(
            audit_id=(
                audit_record.audit_id
                if audit_record is not None
                else ""
            ),
            routing_decision=routing_decision,
            review_recommendation=recommendation,
            final_action=final_action,
            audit_record=audit_record,
        )

    def get_pending_reviews(self) -> List[AuditRecord]:
        """Return all audit records waiting for human review."""

        return self.queue.get_pending()

    def submit_review_decision(
        self,
        audit_id: str,
        decision: ReviewDecision,
        notes: str,
        reviewer: str,
    ) -> bool:
        """
        Submit a human review decision.

        State transition:

            PENDING
                ↓
            IN_REVIEW
                ↓
            RESOLVED

        The audit log remains append-only.
        """

        # Start review.
        success = self.queue.start_review(
            audit_id,
            reviewer,
        )

        if not success:
            return False

        # Resolve the review.
        success = self.queue.resolve(
            audit_id,
            decision,
            notes,
        )

        if not success:
            return False

        # Do not rewrite the immutable historical audit log.
        return True

