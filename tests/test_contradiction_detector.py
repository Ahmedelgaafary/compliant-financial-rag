from src.verification.contradiction_detector import (
    ContradictionDetector,
)
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason


def make_claim(value: str = "Acme Corporation") -> Claim:
    return Claim(
        claim_id="claim-001",
        claim_type=ClaimType.ENTITY,
        subject="Company",
        value=value,
        source_chunk_id="chunk-001",
    )


def test_explicit_not_contradiction() -> None:
    detector = ContradictionDetector()

    result = detector.verify(
        make_claim(),
        "Acme Corporation was not the reporting entity.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.EVIDENCE_CONTRADICTS
    assert result.confidence == 1.0


def test_does_not_contradiction() -> None:
    detector = ContradictionDetector()

    result = detector.verify(
        make_claim(),
        "The filing does not identify Acme Corporation as the entity.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.EVIDENCE_CONTRADICTS


def test_false_contradiction() -> None:
    detector = ContradictionDetector()

    result = detector.verify(
        make_claim(),
        "The statement that Acme Corporation reported the result is false.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.EVIDENCE_CONTRADICTS


def test_denial_contradiction() -> None:
    detector = ContradictionDetector()

    result = detector.verify(
        make_claim(),
        "The company denies Acme Corporation was responsible.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.EVIDENCE_CONTRADICTS


def test_no_contradiction_when_claim_is_supported() -> None:
    detector = ContradictionDetector()

    result = detector.verify(
        make_claim(),
        "Acme Corporation was the reporting entity.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.UNSUPPORTED_CLAIM


def test_no_contradiction_when_claim_is_absent() -> None:
    detector = ContradictionDetector()

    result = detector.verify(
        make_claim(),
        "The filing identifies Globex Corporation.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.UNSUPPORTED_CLAIM


def test_missing_evidence() -> None:
    detector = ContradictionDetector()

    result = detector.verify(
        make_claim(),
        None,
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.EVIDENCE_MISSING
    assert result.confidence == 0.0


def test_empty_evidence() -> None:
    detector = ContradictionDetector()

    result = detector.verify(
        make_claim(),
        "   ",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.EVIDENCE_MISSING


def test_empty_claim_value() -> None:
    detector = ContradictionDetector()

    result = detector.verify(
        make_claim(""),
        "The company denies the claim.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.UNSUPPORTED_CLAIM


def test_case_insensitive_contradiction() -> None:
    detector = ContradictionDetector()

    result = detector.verify(
        make_claim("Acme Corporation"),
        "ACME CORPORATION was not the reporting entity.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.EVIDENCE_CONTRADICTS