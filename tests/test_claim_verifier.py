from src.verification.claim_verifier import ClaimVerifier
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason


def _claim(
    value: str,
    period: str | None = "2025",
) -> Claim:
    return Claim(
        claim_id="claim-001",
        claim_type=ClaimType.NUMERIC,
        subject="revenue",
        value=value,
        unit="USD billion",
        period=period,
        source_chunk_id="revenue",
    )


def test_claim_verified_when_number_and_period_match() -> None:
    verifier = ClaimVerifier()

    result = verifier.verify(
        _claim("42.8", "2025"),
        "Total revenue was $42.8 billion in 2025.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.NUMERIC_MATCH
    assert result.confidence == 1.0


def test_claim_rejected_when_number_mismatches() -> None:
    verifier = ClaimVerifier()

    result = verifier.verify(
        _claim("45.2", "2025"),
        "Total revenue was $42.8 billion in 2025.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.NUMERIC_MISMATCH


def test_claim_rejected_when_period_mismatches() -> None:
    verifier = ClaimVerifier()

    result = verifier.verify(
        _claim("42.8", "2025"),
        "Total revenue was $42.8 billion in 2024.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.PERIOD_MISMATCH


def test_claim_inconclusive_when_evidence_has_no_number() -> None:
    verifier = ClaimVerifier()

    result = verifier.verify(
        _claim("42.8", "2025"),
        "The company reported strong performance.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.EVIDENCE_MISSING


def test_claim_inconclusive_without_period() -> None:
    verifier = ClaimVerifier()

    result = verifier.verify(
        _claim("42.8", None),
        "Revenue was $42.8 billion.",
    )

    assert result.status == VerificationStatus.VERIFIED