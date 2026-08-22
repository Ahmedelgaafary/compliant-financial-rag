from src.verification.models import (
    Claim,
    ClaimType,
    VerificationStatus,
)
from src.verification.numeric_verifier import NumericVerifier
from src.verification.reasons import VerificationReason


def _claim(
    value: str,
    unit: str | None = "USD billion",
) -> Claim:
    return Claim(
        claim_id="claim-001",
        claim_type=ClaimType.NUMERIC,
        subject="revenue",
        value=value,
        unit=unit,
        period="2025",
        source_chunk_id="revenue",
    )


def test_numeric_match() -> None:
    verifier = NumericVerifier()

    result = verifier.verify(
        _claim("42.8"),
        "Total revenue was $42.8 billion in 2025.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.NUMERIC_MATCH
    assert result.confidence == 1.0


def test_numeric_mismatch() -> None:
    verifier = NumericVerifier()

    result = verifier.verify(
        _claim("45.2"),
        "Total revenue was $42.8 billion in 2025.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.NUMERIC_MISMATCH


def test_million_to_absolute_value() -> None:
    verifier = NumericVerifier()

    result = verifier.verify(
        _claim("42.8", "USD million"),
        "Revenue was $42.8 million.",
    )

    assert result.status == VerificationStatus.VERIFIED


def test_comma_separated_number() -> None:
    verifier = NumericVerifier()

    result = verifier.verify(
        _claim("42.8"),
        "Revenue was $42.8 billion.",
    )

    assert result.status == VerificationStatus.VERIFIED


def test_missing_numeric_evidence() -> None:
    verifier = NumericVerifier()

    result = verifier.verify(
        _claim("42.8"),
        "The company reported strong financial performance.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.EVIDENCE_MISSING


def test_unsupported_claim_type() -> None:
    verifier = NumericVerifier()

    claim = Claim(
        claim_id="claim-002",
        claim_type=ClaimType.TEXT,
        subject="revenue",
        value="strong",
    )

    result = verifier.verify(
        claim,
        "Revenue increased significantly.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.UNSUPPORTED_CLAIM


def test_percentage_match() -> None:
    verifier = NumericVerifier()

    result = verifier.verify(
        _claim("5.2", "%"),
        "The default rate increased by 5.2%.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.NUMERIC_MATCH


def test_percentage_mismatch() -> None:
    verifier = NumericVerifier()

    result = verifier.verify(
        _claim("5.2", "%"),
        "The default rate increased by 5.7%.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.NUMERIC_MISMATCH