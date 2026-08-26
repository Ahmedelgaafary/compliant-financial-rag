"""
Unit tests for AgentState – verifying default values, field storage,
and state transition rules.
"""

import pytest

from src.agent.state import (
    AgentState,
    AuditStatus,
    FinalResponseStatus,
    HumanDecision,
)
from src.retrieval.models import RetrievalResult
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)


# Fixtures for minimal objects
@pytest.fixture
def sample_retrieval_result():
    return RetrievalResult(
        chunk_id="chunk1",
        document_id="doc1",
        text="Revenue for 2025 was $42.8 billion.",
        score=0.9,
        page_number=3,
        section="Financials",
        document_sha256="abc123",
        retrieval_method="hybrid_rrf",
    )


@pytest.fixture
def sample_claim():
    return Claim(
        claim_id="claim1",
        claim_type=ClaimType.NUMERIC,
        subject="revenue",
        value="$42.8B",
        unit="billion",
        period="2025",
        source_chunk_id="chunk1",
    )


@pytest.fixture
def sample_verification_result(sample_claim):
    return VerificationResult(
        claim_id=sample_claim.claim_id,
        status=VerificationStatus.VERIFIED,
        reason="numeric_match",
        confidence=0.95,
        evidence_chunk_id="chunk1",
    )


# ------------------------------------------------------------------
# Default State Tests
# ------------------------------------------------------------------

def test_initial_state_defaults():
    state = AgentState()
    assert state.user_query == ""
    assert state.retrieval_results == []
    assert state.claims == []
    assert state.verification_results == []
    assert state.risk_assessment is None
    assert state.audit_status == AuditStatus.PENDING
    assert state.human_decision is None
    assert state.final_response_status == FinalResponseStatus.NOT_STARTED


# ------------------------------------------------------------------
# Evidence / Provenance Preservation
# ------------------------------------------------------------------

def test_retrieval_results_preserve_provenance(sample_retrieval_result):
    state = AgentState()
    state.retrieval_results = [sample_retrieval_result]
    assert len(state.retrieval_results) == 1
    result = state.retrieval_results[0]
    assert result.document_id == "doc1"
    assert result.chunk_id == "chunk1"
    assert result.document_sha256 == "abc123"
    assert result.page_number == 3


# ------------------------------------------------------------------
# Verification Results Storage
# ------------------------------------------------------------------

def test_verification_results_store_status(sample_verification_result):
    state = AgentState()
    state.verification_results = [sample_verification_result]
    assert state.verification_results[0].status == VerificationStatus.VERIFIED
    assert state.verification_results[0].evidence_chunk_id == "chunk1"


# ------------------------------------------------------------------
# Risk Decision Storage
# ------------------------------------------------------------------

def test_risk_assessment_can_be_stored():
    state = AgentState()
    state.risk_assessment = None  # placeholder – will set later in integration
    # In practice, risk_assessment is set by guardrails node.
    # We can just assert that the field exists and can be assigned.
    state.risk_assessment = "high"  # simplified for test
    assert state.risk_assessment == "high"


# ------------------------------------------------------------------
# Audit Status Transitions
# ------------------------------------------------------------------

def test_audit_status_can_be_changed():
    state = AgentState()
    assert state.audit_status == AuditStatus.PENDING
    state.set_audit_status(AuditStatus.IN_REVIEW)
    assert state.audit_status == AuditStatus.IN_REVIEW


# ------------------------------------------------------------------
# Human Decision Transitions
# ------------------------------------------------------------------

def test_cannot_set_human_decision_before_review():
    state = AgentState()
    with pytest.raises(ValueError):
        state.set_human_decision(HumanDecision.APPROVED)


def test_set_human_decision_after_review():
    state = AgentState()
    state.set_audit_status(AuditStatus.IN_REVIEW)
    state.set_human_decision(HumanDecision.APPROVED)
    assert state.human_decision == HumanDecision.APPROVED


# ------------------------------------------------------------------
# Final Response Status
# ------------------------------------------------------------------

def test_final_response_status_transition():
    state = AgentState()
    assert state.final_response_status == FinalResponseStatus.NOT_STARTED
    state.set_final_response_status(FinalResponseStatus.GENERATED)
    assert state.final_response_status == FinalResponseStatus.GENERATED