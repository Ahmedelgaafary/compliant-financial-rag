"""
Integration tests for risk engine → audit routing.
Verifies that HIGH risk cases are queued, LOW risk cases bypass queue,
and that evidence/provenance are preserved.
"""


import pytest

from src.audit.models import AuditStatus
from src.audit.review_service import ReviewService
from src.guardrails.policies import GuardrailPolicies
from src.guardrails.risk_engine import RiskAssessment, RiskEngine
from src.retrieval.models import RetrievalResult
from src.verification.models import (
    VerificationResult,
    VerificationStatus,
)


def make_retrieval_result(chunk_id="chunk1", document_id="doc1", score=0.9, page=3):
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        text="Revenue for 2025 was $42.8 billion.",
        score=score,
        page_number=page,
        section="Financials",
        document_sha256="abc123",
        retrieval_method="hybrid_rrf",
    )


def make_verification_result(
    status,
    reason,
    confidence=1.0,
    evidence_chunk_id="chunk1"):
    return VerificationResult(
        claim_id="claim1",
        status=status,
        reason=reason,
        confidence=confidence,
        evidence_chunk_id=evidence_chunk_id,
    )


def make_risk_assessment(risk_score, risk_level, recommended_action, triggers=None):
    return RiskAssessment(
        risk_score=risk_score,
        risk_level=risk_level,
        triggers=triggers or [],
        recommended_action=recommended_action,
    )


@pytest.fixture
def policies():
    return GuardrailPolicies()


@pytest.fixture
def risk_engine(policies):
    return RiskEngine(policies)


@pytest.fixture
def review_service():
    # Use a fresh ReviewService for each test to isolate queue state
    return ReviewService()


def test_low_risk_does_not_create_audit_record(review_service):
    """LOW risk should not create an audit record (should_create_audit_record=False)."""
    risk = make_risk_assessment(
        risk_score=0.1,
        risk_level="LOW",
        recommended_action="AUTO_ANSWER",
    )
    retrieval = [make_retrieval_result()]
    verification = [
        make_verification_result(VerificationStatus.VERIFIED, "numeric_match")
    ]

    outcome = review_service.initiate_review(
        user_query="What was revenue?",
        claim="Revenue was $42.8B",
        verification_status="VERIFIED",
        verification_reason="numeric_match",
        risk_assessment=risk,
        verification_results=verification,
        evidence=[{"text": r.text, "page": r.page_number} for r in retrieval],
        document_id=retrieval[0].document_id,
        document_sha256=retrieval[0].document_sha256,
        page_number=retrieval[0].page_number,
    )

    # Since risk is LOW and should_create_audit_record=False, 
    #audit_record should be None
    assert outcome.audit_record is None
    assert outcome.final_action == "AUTO_ANSWER"


def test_high_risk_creates_audit_record(review_service):
    """HIGH risk should create an audit record and enqueue it."""
    risk = make_risk_assessment(
        risk_score=0.9,
        risk_level="HIGH",
        recommended_action="HUMAN_REVIEW",
        triggers=["NUMERIC_MISMATCH"],
    )
    retrieval = [make_retrieval_result()]
    verification = [
        make_verification_result(VerificationStatus.REJECTED, "numeric_mismatch")
    ]

    outcome = review_service.initiate_review(
        user_query="What was revenue?",
        claim="Revenue was $42.8B",
        verification_status="REJECTED",
        verification_reason="numeric_mismatch",
        risk_assessment=risk,
        verification_results=verification,
        evidence=[{"text": r.text, "page": r.page_number} for r in retrieval],
        document_id=retrieval[0].document_id,
        document_sha256=retrieval[0].document_sha256,
        page_number=retrieval[0].page_number,
    )

    assert outcome.audit_record is not None
    assert outcome.final_action == "HUMAN_REVIEW"
    assert outcome.audit_record.status == AuditStatus.PENDING

    # Verify the queue contains this record
    pending = review_service.get_pending_reviews()
    assert len(pending) == 1
    assert pending[0].audit_id == outcome.audit_id


def test_evidence_and_provenance_preserved(review_service):
    """When a review record is created, evidence and provenance must be preserved."""
    risk = make_risk_assessment(
        risk_score=0.8,
        risk_level="HIGH",
        recommended_action="HUMAN_REVIEW",
        triggers=["NUMERIC_MISMATCH"],
    )
    retrieval = [
        make_retrieval_result(chunk_id="chunk1", document_id="doc1", page=3),
        make_retrieval_result(chunk_id="chunk2", document_id="doc1", page=5),
    ]
    verification = [
        make_verification_result(VerificationStatus.REJECTED, "numeric_mismatch")
    ]

    outcome = review_service.initiate_review(
        user_query="What was revenue?",
        claim="Revenue was $42.8B",
        verification_status="REJECTED",
        verification_reason="numeric_mismatch",
        risk_assessment=risk,
        verification_results=verification,
        evidence=[{"text": r.text, "page": r.page_number} for r in retrieval],
        document_id=retrieval[0].document_id,
        document_sha256=retrieval[0].document_sha256,
        page_number=retrieval[0].page_number,
    )

    record = outcome.audit_record
    assert record is not None

    # Evidence list preserved
    assert len(record.evidence) == 2
    assert record.evidence[0]["text"] == retrieval[0].text
    assert record.evidence[0]["page"] == 3

    # Provenance fields preserved
    assert record.document_id == "doc1"
    assert record.document_sha256 == "abc123"
    assert record.page_number == 3

    # Verification results preserved (as list of dicts)
    assert len(record.verification_results) == 1
    assert record.verification_results[0]["status"] == "rejected"


def test_high_risk_cannot_bypass_review(review_service):
    """Even if risk level is HIGH, the queue must contain the record."""
    risk = make_risk_assessment(
        risk_score=0.95,
        risk_level="HIGH",
        recommended_action="HUMAN_REVIEW",
        triggers=["NUMERIC_MISMATCH"],
    )
    # No retrieval results (simulate missing evidence)
    verification = [
        make_verification_result(VerificationStatus.REJECTED, "numeric_mismatch")
    ]

    outcome = review_service.initiate_review(
        user_query="What was revenue?",
        claim="Revenue was $42.8B",
        verification_status="REJECTED",
        verification_reason="numeric_mismatch",
        risk_assessment=risk,
        verification_results=verification,
        evidence=[],
        document_id="",
        document_sha256="",
        page_number=1,
    )

    assert outcome.audit_record is not None
    # The router should always route HIGH to review
    assert outcome.routing_decision.action.value == "HUMAN_REVIEW"
    pending = review_service.get_pending_reviews()
    assert len(pending) == 1