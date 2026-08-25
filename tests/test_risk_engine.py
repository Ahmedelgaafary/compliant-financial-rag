"""
Unit tests for the deterministic risk engine.
Run with: pytest tests/test_risk_engine.py -v
"""

import pytest

from src.guardrails.policies import GuardrailPolicies
from src.guardrails.risk_engine import RiskEngine
from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult, VerificationStatus


@pytest.fixture
def policies():
    return GuardrailPolicies()


@pytest.fixture
def engine(policies):
    return RiskEngine(policies)


@pytest.fixture
def confidence_high():
    # If the actual ConfidenceScore class expects other fields, adapt.
    # For simplicity, we use a dummy object with an 'overall' attribute.
    class DummyConfidence:
        overall = 0.9
    return DummyConfidence()


@pytest.fixture
def confidence_low():
    class DummyConfidence:
        overall = 0.5
    return DummyConfidence()


def make_verification_result(
    claim_id="claim1",
    status=VerificationStatus.VERIFIED,
    reason="numeric_match",
    confidence=1.0,
    evidence_chunk_id="chunk1",
) -> VerificationResult:
    return VerificationResult(
        claim_id=claim_id,
        status=status,
        reason=reason,
        confidence=confidence,
        evidence_chunk_id=evidence_chunk_id,
    )


def make_retrieval_results(count=1):
    return [
        RetrievalResult(
            chunk_id=f"chunk{i}",
            document_id="doc1",
            text="Evidence text",
            score=0.9,
            page_number=1,
            section="Financials",
            document_sha256="abc",
            retrieval_method="hybrid_rrf",
        )
        for i in range(count)
    ]


# --- Tests ---

def test_verified_claim_low_risk(engine, policies, confidence_high):
    verification = [
        make_verification_result(status=VerificationStatus.VERIFIED)
    ]
    retrieval = make_retrieval_results(5)
    assessment = engine.assess(retrieval, verification, confidence_high)
    assert assessment.risk_level == "LOW"
    assert assessment.recommended_action == "AUTO_ANSWER"
    assert assessment.risk_score < policies.risk_medium_threshold


def test_inconclusive_claim_increases_risk(engine, policies, confidence_high):
    verification = [
        make_verification_result(
            status=VerificationStatus.INCONCLUSIVE,
            reason="evidence_missing"
        )
    ]
    retrieval = make_retrieval_results(5)
    assessment = engine.assess(retrieval, verification, confidence_high)
    assert assessment.risk_score > 0
    assert "INCONCLUSIVE" in " ".join(assessment.triggers)


def test_rejected_claim_high_risk(engine, policies, confidence_high):
    verification = [
        make_verification_result(
            status=VerificationStatus.REJECTED,
            reason="numeric_mismatch"
        )
    ]
    retrieval = make_retrieval_results(5)
    assessment = engine.assess(retrieval, verification, confidence_high)
    assert assessment.risk_level == "HIGH"
    assert assessment.recommended_action == "HUMAN_REVIEW"
    assert "NUMERIC_MISMATCH" in assessment.triggers


def test_numeric_mismatch_blocks(engine, policies, confidence_high):
    verification = [
        make_verification_result(
            status=VerificationStatus.REJECTED,
            reason="numeric_mismatch"
        )
    ]
    retrieval = make_retrieval_results(5)
    # Enable block_on_numeric_mismatch (should be True by default)
    policies.block_on_numeric_mismatch = True
    assessment = engine.assess(retrieval, verification, confidence_high)
    assert assessment.recommended_action == "BLOCK"


def test_contradiction_escalates(engine, policies, confidence_high):
    verification = [
        make_verification_result(
            status=VerificationStatus.REJECTED,
            reason="evidence_contradicts"
        ),
        make_verification_result(
            status=VerificationStatus.REJECTED,
            reason="evidence_contradicts"
        ),
    ]
    retrieval = make_retrieval_results(5)
    assessment = engine.assess(retrieval, verification, confidence_high)
    assert "EVIDENCE_CONTRADICTS" in assessment.triggers
    assert assessment.risk_score >= 0.5


def test_missing_provenance_raises_risk(engine, policies, confidence_high):
    verification = [
        make_verification_result(
            status=VerificationStatus.VERIFIED,
            evidence_chunk_id=None
        )
    ]
    retrieval = make_retrieval_results(5)
    assessment = engine.assess(retrieval, verification, confidence_high)
    assert "MISSING_PROVENANCE" in assessment.triggers
    assert assessment.recommended_action != "AUTO_ANSWER"


def test_low_confidence_raises_risk(engine, policies, confidence_low):
    verification = [
        make_verification_result(status=VerificationStatus.VERIFIED)
    ]
    retrieval = make_retrieval_results(5)
    assessment = engine.assess(retrieval, verification, confidence_low)
    assert "LOW_CONFIDENCE" in assessment.triggers
    assert assessment.risk_level in ["MEDIUM", "HIGH"]


def test_no_evidence_blocks(engine, policies, confidence_high):
    verification = []
    retrieval = []
    assessment = engine.assess(retrieval, verification, confidence_high)
    assert assessment.risk_score >= 0.8
    assert assessment.recommended_action == "HUMAN_REVIEW"


def test_deterministic_same_inputs(engine, policies, confidence_high):
    verification = [
        make_verification_result(status=VerificationStatus.INCONCLUSIVE)
    ]
    retrieval = make_retrieval_results(3)
    assessment1 = engine.assess(retrieval, verification, confidence_high)
    assessment2 = engine.assess(retrieval, verification, confidence_high)
    assert assessment1 == assessment2