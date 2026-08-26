from typing import Optional

from src.agent.state import AgentState
from src.audit.models import AuditRecord
from src.audit.review_service import ReviewService


class AuditAgent:
    """Uses the ReviewService to create audit records."""

    def __init__(self, review_service: ReviewService = None):
        self.review_service = review_service or ReviewService()

    def initiate_review(self, state: AgentState) -> Optional[AuditRecord]:
        evidence = [
            {"text": r.text, "page": r.page_number, "chunk_id": r.chunk_id}
            for r in state.retrieval_results
        ]
        first_verification = (
            state.verification_results[0] if state.verification_results else None
        )
        verification_status = (
            first_verification.status.value if first_verification else "inconclusive"
        )
        verification_reason = (
            first_verification.reason if first_verification else "EVIDENCE_MISSING"
        )
        first_result = state.retrieval_results[0] if state.retrieval_results else None

        outcome = self.review_service.initiate_review(
            user_query=state.user_query,
            claim=state.raw_llm_output or "No claim extracted",
            verification_status=verification_status,
            verification_reason=verification_reason,
            risk_assessment=state.risk_assessment,
            verification_results=state.verification_results,
            evidence=evidence,
            document_id=first_result.document_id if first_result else "",
            document_sha256=first_result.document_sha256 if first_result else "",
            page_number=first_result.page_number if first_result else 1,
        )
        return outcome.audit_record