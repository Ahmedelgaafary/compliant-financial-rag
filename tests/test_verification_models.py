from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason


def test_claim_creation() -> None:
    claim = Claim(
        claim_id="claim-001",
        claim_type=ClaimType.NUMERIC,
        subject="revenue",
        value="42.8",
        unit="USD billion",
        period="2025",
        source_chunk_id="revenue",
    )

    assert claim.claim_id == "claim-001"
    assert claim.claim_type == ClaimType.NUMERIC
    assert claim.subject == "revenue"
    assert claim.value == "42.8"
    assert claim.unit == "USD billion"
    assert claim.period == "2025"
    assert claim.source_chunk_id == "revenue"


def test_verification_result() -> None:
    result = VerificationResult(
        claim_id="claim-001",
        status=VerificationStatus.VERIFIED,
        reason=VerificationReason.NUMERIC_MATCH,
        confidence=1.0,
        evidence_chunk_id="revenue",
    )

    assert result.claim_id == "claim-001"
    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.NUMERIC_MATCH
    assert result.confidence == 1.0
    assert result.evidence_chunk_id == "revenue"


def test_verification_status_values() -> None:
    assert VerificationStatus.VERIFIED.value == "verified"
    assert VerificationStatus.REJECTED.value == "rejected"
    assert VerificationStatus.INCONCLUSIVE.value == "inconclusive"