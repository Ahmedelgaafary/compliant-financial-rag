from src.verification.date_verifier import DateVerifier
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason


def make_date_claim(
    value: str = "2025-12-31",
) -> Claim:
    return Claim(
        claim_id="claim-001",
        claim_type=ClaimType.DATE,
        subject="Report date",
        value=value,
        source_chunk_id="chunk-001",
    )


def test_iso_date_matches() -> None:
    verifier = DateVerifier()

    result = verifier.verify(
        make_date_claim("2025-12-31"),
        "The report was published on 2025-12-31.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.PERIOD_MATCH
    assert result.confidence == 1.0


def test_us_date_matches() -> None:
    verifier = DateVerifier()

    result = verifier.verify(
        make_date_claim("12/31/2025"),
        "The report was published on 12/31/2025.",
    )

    assert result.status == VerificationStatus.VERIFIED


def test_european_date_matches() -> None:
    verifier = DateVerifier()

    result = verifier.verify(
        make_date_claim("31/12/2025"),
        "The report was published on 31/12/2025.",
    )

    assert result.status == VerificationStatus.VERIFIED


def test_long_month_date_matches() -> None:
    verifier = DateVerifier()

    result = verifier.verify(
        make_date_claim("December 31, 2025"),
        "The report was published on December 31, 2025.",
    )

    assert result.status == VerificationStatus.VERIFIED


def test_day_month_year_matches() -> None:
    verifier = DateVerifier()

    result = verifier.verify(
        make_date_claim("31 December 2025"),
        "The report was published on 31 December 2025.",
    )

    assert result.status == VerificationStatus.VERIFIED


def test_date_mismatch() -> None:
    verifier = DateVerifier()

    result = verifier.verify(
        make_date_claim("2025-12-31"),
        "The report was published on 2025-11-30.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.PERIOD_MISMATCH


def test_missing_evidence() -> None:
    verifier = DateVerifier()

    result = verifier.verify(
        make_date_claim(),
        None,
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.EVIDENCE_MISSING


def test_empty_evidence() -> None:
    verifier = DateVerifier()

    result = verifier.verify(
        make_date_claim(),
        "   ",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.EVIDENCE_MISSING


def test_invalid_date_is_unsupported() -> None:
    verifier = DateVerifier()

    result = verifier.verify(
        make_date_claim("not-a-date"),
        "The report was published on 2025-12-31.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.UNSUPPORTED_CLAIM


def test_non_date_claim_is_unsupported() -> None:
    verifier = DateVerifier()

    claim = Claim(
        claim_id="claim-002",
        claim_type=ClaimType.NUMERIC,
        subject="Revenue",
        value="100",
    )

    result = verifier.verify(
        claim,
        "Revenue was 100 million on 2025-12-31.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.UNSUPPORTED_CLAIM


def test_evidence_without_dates_is_inconclusive() -> None:
    verifier = DateVerifier()

    result = verifier.verify(
        make_date_claim(),
        "The company reported strong annual performance.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.EVIDENCE_MISSING


def test_evidence_chunk_id_is_preserved() -> None:
    verifier = DateVerifier()

    result = verifier.verify(
        make_date_claim(),
        "The report was published on 2025-12-31.",
    )

    assert result.evidence_chunk_id == "chunk-001"