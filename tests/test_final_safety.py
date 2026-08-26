import pytest

from src.guardrails.final_safety import FinalSafetyValidator
from src.guardrails.risk_engine import RiskAssessment
from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult, VerificationStatus


def make_retrieval_result(chunk_id="chunk1", document_id="doc1", sha256="sha1"):
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        text="Evidence text",
        score=0.9,
        page_number=1,
        section="Financials",
        document_sha256=sha256,
        retrieval_method="hybrid_rrf",
    )


def make_verification_result(status, reason="numeric_match"):
    return VerificationResult(
        claim_id="claim1",
        status=status,
        reason=reason,
        confidence=1.0,
        evidence_chunk_id="chunk1",
    )


def make_risk(level="LOW", action="AUTO_ANSWER"):
    return RiskAssessment(
        risk_score=0.1,
        risk_level=level,
        triggers=[],
        recommended_action=action,
    )


@pytest.fixture
def validator():
    return FinalSafetyValidator()


def test_allows_valid_low_risk(validator):
    result = validator.validate(
        "Revenue is $42.8B",
        verification_results=[make_verification_result(VerificationStatus.VERIFIED)],
        retrieval_results=[make_retrieval_result()],
        risk_assessment=make_risk(),
        confidence_score=0.9,
    )
    assert result.allowed is True
    assert result.reasons == []


def test_blocks_missing_evidence(validator):
    result = validator.validate(
        "Revenue is $42.8B",
        verification_results=[make_verification_result(VerificationStatus.VERIFIED)],
        retrieval_results=[],
        risk_assessment=make_risk(),
        confidence_score=0.9,
    )
    assert result.allowed is False
    assert "NO_EVIDENCE" in result.reasons


def test_blocks_missing_provenance(validator):
    retrieval = RetrievalResult(
        chunk_id="chunk1",
        document_id="doc1",
        text="Evidence text",
        score=0.9,
        page_number=1,
        section="Financials",
        document_sha256="",  # missing hash
        retrieval_method="hybrid_rrf",
    )
    result = validator.validate(
        "Revenue is $42.8B",
        verification_results=[make_verification_result(VerificationStatus.VERIFIED)],
        retrieval_results=[retrieval],
        risk_assessment=make_risk(),
        confidence_score=0.9,
    )
    assert result.allowed is False
    assert any("MISSING_PROVENANCE" in r for r in result.reasons)

def test_blocks_contradiction(validator):
    verification = make_verification_result(
        VerificationStatus.REJECTED,
        "evidence_contradicts",
    )
    result = validator.validate(
        "Revenue is $42.8B",
        verification_results=[verification],
        retrieval_results=[make_retrieval_result()],
        risk_assessment=make_risk(),
        confidence_score=0.9,
    )
    assert result.allowed is False
    assert "UNRESOLVED_CONTRADICTION" in result.reasons

def test_blocks_high_risk(validator):
    risk = make_risk(level="HIGH", action="HUMAN_REVIEW")
    result = validator.validate(
        "Revenue is $42.8B",
        verification_results=[make_verification_result(VerificationStatus.INCONCLUSIVE)],
        retrieval_results=[make_retrieval_result()],
        risk_assessment=risk,
        confidence_score=0.9,
    )
    assert result.allowed is False
    assert any("RISK_POLICY_BLOCKS_AUTO_ANSWER" in r for r in result.reasons)


def test_blocks_low_confidence(validator):
    result = validator.validate(
        "Revenue is $42.8B",
        verification_results=[make_verification_result(VerificationStatus.VERIFIED)],
        retrieval_results=[make_retrieval_result()],
        risk_assessment=make_risk(),
        confidence_score=0.5,
    )
    assert result.allowed is False
    assert "LOW_CONFIDENCE" in result.reasons