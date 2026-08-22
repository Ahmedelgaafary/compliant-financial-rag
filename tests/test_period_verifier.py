from src.verification.models import (
    Claim,
    ClaimType,
    VerificationStatus,
)
from src.verification.period_verifier import PeriodVerifier
from src.verification.reasons import VerificationReason


def _claim(period: str | None) -> Claim:
    return Claim(
        claim_id="claim-period-001",
        claim_type=ClaimType.NUMERIC,
        subject="revenue",
        value="42.8",
        unit="USD billion",
        period=period,
        source_chunk_id="revenue",
    )


def test_period_match() -> None:
    verifier = PeriodVerifier()

    result = verifier.verify(
        _claim("2025"),
        "Total revenue was $42.8 billion in 2025.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.PERIOD_MATCH


def test_period_mismatch() -> None:
    verifier = PeriodVerifier()

    result = verifier.verify(
        _claim("2025"),
        "Total revenue was $42.8 billion in 2024.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.PERIOD_MISMATCH


def test_missing_period_evidence() -> None:
    verifier = PeriodVerifier()

    result = verifier.verify(
        _claim("2025"),
        "Total revenue was $42.8 billion.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.EVIDENCE_MISSING


def test_missing_claim_period() -> None:
    verifier = PeriodVerifier()

    result = verifier.verify(
        _claim(None),
        "Total revenue was $42.8 billion in 2025.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.UNSUPPORTED_CLAIM


def test_invalid_claim_period() -> None:
    verifier = PeriodVerifier()

    result = verifier.verify(
        _claim("FY-UNKNOWN"),
        "Total revenue was $42.8 billion in 2025.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.UNSUPPORTED_CLAIM


def test_multiple_evidence_years() -> None:
    verifier = PeriodVerifier()

    result = verifier.verify(
        _claim("2025"),
        (
            "Revenue was $38.1 billion in 2024 "
            "and $42.8 billion in 2025."
        ),
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.PERIOD_MATCH