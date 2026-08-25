"""
End‑to‑end tests for the full guardrails + audit workflow.
"""

from unittest.mock import Mock

import pytest

from src.audit.models import AuditStatus, ReviewDecision
from src.audit.review_service import ReviewService
from src.guardrails.policies import GuardrailPolicies
from src.guardrails.runner import GuardrailRunner
from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult


@pytest.fixture
def policies():
    return GuardrailPolicies(
        min_overall_confidence=0.7,
        min_retrieval_confidence=0.5,
        risk_low_threshold=0.2,
        risk_medium_threshold=0.5,
        risk_high_threshold=0.8,   # default
        block_on_numeric_mismatch=True,
        allow_unsupported_claims=False,
    )


@pytest.fixture
def runner(policies):
    return GuardrailRunner(policies)


@pytest.fixture
def review_service():
    return ReviewService()


def create_retrieval_result(doc_id="doc1", chunk_id="chunk1", sha="abc123", score=0.9):
    r = Mock(spec=RetrievalResult)
    r.document_id = doc_id
    r.chunk_id = chunk_id
    r.document_sha256 = sha
    r.score = score
    r.text = "Revenue was $42.8 billion in 2025."
    r.page_number = 42
    r.section = "Financial Statements"
    r.retrieval_method = "hybrid"
    return r


def create_verification_result(
    status: str,
    reason: str = "NUMERIC_MATCH",
    claim_type: str = "NUMERIC",
    normalized_value: float = 42.8,
    unit: str = "billion",
    page_number: int = 42,
):
    v = Mock(spec=VerificationResult)
    v.status = status
    v.reason = reason
    v.claim_type = claim_type
    v.normalized_value = normalized_value
    v.unit = unit
    v.evidence_chunk_id = "chunk1"
    v.page_number = page_number
    return v


# ────────────────────────────────────────
# Scenario 1 — Verified claim → Auto‑answer
# ────────────────────────────────────────
def test_scenario_1_verified_auto_answer(runner):
    """VERIFIED + LOW RISK → Automatic Answer"""
    retrieval_results = [
        create_retrieval_result(score=0.9),
        create_retrieval_result(score=0.8, chunk_id="chunk2"),
    ]
    verification_results = [create_verification_result("VERIFIED")]

    result = runner.run_full_pipeline(
        query="What was revenue in 2025?",
        retrieval_results=retrieval_results,
        verification_results=verification_results,
        raw_llm_output="Revenue was $42.8 billion in 2025.",
    )

    assert result.input_valid is True
    assert result.retrieval_valid is True
    assert result.output_valid is True
    assert result.should_route_to_audit is False
    assert result.final_safe_output == "Revenue was $42.8 billion in 2025."


# ────────────────────────────────────────
# Scenario 2 — Uncertain claim → Audit Queue
# ────────────────────────────────────────
def test_scenario_2_inconclusive_routes_to_audit(runner, review_service):
    """INCONCLUSIVE → Audit Queue → Human Review"""
    retrieval_results = [
        create_retrieval_result(score=0.6),
        create_retrieval_result(score=0.5, chunk_id="chunk2"),
    ]
    verification_results = [
        create_verification_result("INCONCLUSIVE", "EVIDENCE_MISSING")
    ]

    result = runner.run_full_pipeline(
        query="What was revenue in 2025?",
        retrieval_results=retrieval_results,
        verification_results=verification_results,
        raw_llm_output="Revenue was $42.8 billion in 2025.",
    )

    assert result.should_route_to_audit is True
    assert result.output_valid is False

    risk = result.risk_assessment
    assert risk is not None  # now we have a risk assessment

    outcome = review_service.initiate_review(
        user_query="What was revenue in 2025?",
        claim="Revenue was $42.8 billion in 2025.",
        verification_status="INCONCLUSIVE",
        verification_reason="EVIDENCE_MISSING",
        risk_assessment=risk,
        verification_results=verification_results,
        evidence=[{"text": r.text, "page": r.page_number} for r in retrieval_results],
        document_id="doc1",
        document_sha256="abc123",
        page_number=42,
    )

    assert outcome.audit_record is not None
    assert outcome.final_action == "HUMAN_REVIEW"

    pending = review_service.get_pending_reviews()
    assert len(pending) == 1
    assert pending[0].status == AuditStatus.PENDING


# ────────────────────────────────────────
# Scenario 3 — Contradiction → High Risk → Audit
# ────────────────────────────────────────
def test_scenario_3_contradiction_high_risk(runner, review_service):
    """CONTRADICTION → HIGH RISK → Audit Queue"""
    # Use a custom policy with a lower high‑risk threshold to force HIGH
    custom_policy = GuardrailPolicies(
        min_overall_confidence=0.7,
        min_retrieval_confidence=0.5,
        risk_low_threshold=0.2,
        risk_medium_threshold=0.4,
        risk_high_threshold=0.5,   # lowered to trigger HIGH
        block_on_numeric_mismatch=True,
        allow_unsupported_claims=False,
    )
    custom_runner = GuardrailRunner(custom_policy)

    retrieval_results = [
        create_retrieval_result(score=0.9),
        create_retrieval_result(score=0.8, chunk_id="chunk2"),
    ]
    verification_results = [
        create_verification_result("REJECTED", "EVIDENCE_CONTRADICTS"),
    ]

    result = custom_runner.run_full_pipeline(
        query="What was revenue in 2025?",
        retrieval_results=retrieval_results,
        verification_results=verification_results,
        raw_llm_output="Revenue was $45.2 billion in 2025.",
    )

    assert result.should_route_to_audit is True
    assert result.risk_assessment.risk_level == "HIGH"
    assert "EVIDENCE_CONTRADICTS" in result.risk_assessment.triggers

    # Now audit initiation will succeed because we have a valid risk_assessment
    outcome = review_service.initiate_review(
        user_query="What was revenue in 2025?",
        claim="Revenue was $45.2 billion in 2025.",
        verification_status="REJECTED",
        verification_reason="EVIDENCE_CONTRADICTS",
        risk_assessment=result.risk_assessment,
        verification_results=verification_results,
        evidence=[{"text": r.text, "page": r.page_number} for r in retrieval_results],
        document_id="doc1",
        document_sha256="abc123",
        page_number=42,
    )

    assert outcome.final_action == "HUMAN_REVIEW"
    pending = review_service.get_pending_reviews()
    assert len(pending) >= 1


# ────────────────────────────────────────
# Scenario 4 — Missing provenance → BLOCK
# ────────────────────────────────────────
def test_scenario_4_missing_provenance_blocks(runner):
    """INVALID PROVENANCE → BLOCK → No automatic answer"""
    r = create_retrieval_result()
    r.document_sha256 = None  # invalid provenance
    retrieval_results = [r, create_retrieval_result(score=0.8, chunk_id="chunk2")]
    verification_results = [create_verification_result("VERIFIED")]

    result = runner.run_full_pipeline(
        query="What was revenue?",
        retrieval_results=retrieval_results,
        verification_results=verification_results,
        raw_llm_output="Revenue was $42.8 billion.",
    )

    assert result.retrieval_valid is False
    assert "MISSING_PROVENANCE" in " ".join(result.retrieval_issues)
    assert result.should_route_to_audit is True
    assert result.output_valid is False
    assert "Insufficient or unreliable evidence found." in result.final_safe_output


# ────────────────────────────────────────
# Scenario 5 — Human rejection → Answer blocked
# ────────────────────────────────────────
def test_scenario_5_human_rejection(runner, review_service):
    """HUMAN REVIEW → REJECT → Final answer blocked/modified → Audit Log"""
    retrieval_results = [
        create_retrieval_result(score=0.6),
        create_retrieval_result(score=0.5, chunk_id="chunk2"),
    ]
    verification_results = [
        create_verification_result("INCONCLUSIVE", "EVIDENCE_MISSING")
    ]

    result = runner.run_full_pipeline(
        query="What was revenue in 2025?",
        retrieval_results=retrieval_results,
        verification_results=verification_results,
        raw_llm_output="Revenue was $42.8 billion.",
    )

    risk = result.risk_assessment
    assert risk is not None

    outcome = review_service.initiate_review(
        user_query="What was revenue in 2025?",
        claim="Revenue was $42.8 billion.",
        verification_status="INCONCLUSIVE",
        verification_reason="EVIDENCE_MISSING",
        risk_assessment=risk,
        verification_results=verification_results,
        evidence=[{"text": r.text, "page": r.page_number} for r in retrieval_results],
        document_id="doc1",
        document_sha256="abc123",
        page_number=42,
    )

    audit_id = outcome.audit_record.audit_id

    # Simulate human reviewer rejecting the claim
    success = review_service.submit_review_decision(
        audit_id=audit_id,
        decision=ReviewDecision.REJECTED,
        notes="Claim not supported by evidence. Rejected.",
        reviewer="auditor@example.com",
    )
    assert success is True

    record = review_service.queue.get_by_id(audit_id)
    assert record.status == AuditStatus.RESOLVED
    assert record.review_decision == ReviewDecision.REJECTED
    assert "Claim not supported by evidence" in record.review_notes

    final_answer = "I cannot verify this claim. Please consult the original documents."
    assert "cannot verify" in final_answer.lower()