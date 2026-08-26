from src.verification.entity_verifier import EntityVerifier
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason


def make_entity_claim(
    value: str = "Acme Corporation",
) -> Claim:
    return Claim(
        claim_id="claim-001",
        claim_type=ClaimType.ENTITY,
        subject="Company",
        value=value,
        source_chunk_id="source-001",
    )


def test_entity_match() -> None:
    verifier = EntityVerifier()

    result = verifier.verify(
        make_entity_claim(),
        "Acme Corporation reported strong annual revenue.",
        evidence_chunk_id="evidence-001",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.ENTITY_MATCH
    assert result.confidence == 1.0
    assert result.evidence_chunk_id == "evidence-001"


def test_entity_match_is_case_insensitive() -> None:
    verifier = EntityVerifier()

    result = verifier.verify(
        make_entity_claim(),
        "ACME CORPORATION reported strong annual revenue.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.ENTITY_MATCH


def test_entity_match_normalizes_whitespace() -> None:
    verifier = EntityVerifier()

    result = verifier.verify(
        make_entity_claim("Acme   Corporation"),
        "Acme Corporation reported strong annual revenue.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.ENTITY_MATCH


def test_entity_mismatch() -> None:
    verifier = EntityVerifier()

    result = verifier.verify(
        make_entity_claim(),
        "Globex Corporation reported strong annual revenue.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.ENTITY_MISMATCH
    assert result.confidence == 1.0


def test_missing_evidence() -> None:
    verifier = EntityVerifier()

    result = verifier.verify(
        make_entity_claim(),
        None,
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.EVIDENCE_MISSING
    assert result.confidence == 0.0


def test_empty_evidence() -> None:
    verifier = EntityVerifier()

    result = verifier.verify(
        make_entity_claim(),
        "   ",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.EVIDENCE_MISSING


def test_non_entity_claim_is_unsupported() -> None:
    verifier = EntityVerifier()

    claim = Claim(
        claim_id="claim-002",
        claim_type=ClaimType.NUMERIC,
        subject="Revenue",
        value="100",
    )

    result = verifier.verify(
        claim,
        "Revenue was 100 million.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.UNSUPPORTED_CLAIM
    assert result.confidence == 0.0


def test_empty_entity_value_is_rejected() -> None:
    verifier = EntityVerifier()

    result = verifier.verify(
        make_entity_claim(""),
        "Acme Corporation reported strong annual revenue.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.ENTITY_MISMATCH