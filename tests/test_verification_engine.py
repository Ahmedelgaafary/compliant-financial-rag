from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason
from src.verification.verification_engine import (
    VerificationEngine,
)


class FakeVerifier:
    def __init__(
        self,
        reason: VerificationReason,
    ) -> None:
        self.reason = reason
        self.calls = 0

    def verify(
        self,
        claim: Claim,
        evidence: str,
    ) -> VerificationResult:
        self.calls += 1

        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.VERIFIED,
            reason=self.reason,
            confidence=1.0,
            evidence_chunk_id=claim.source_chunk_id,
        )


def make_claim(
    claim_type: ClaimType,
    period: str | None = None,
) -> Claim:
    return Claim(
        claim_id="claim-001",
        claim_type=claim_type,
        subject="Revenue",
        value="42.8",
        period=period,
        source_chunk_id="chunk-001",
    )


def test_numeric_claim_uses_numeric_verifier() -> None:
    verifier = FakeVerifier(
        VerificationReason.NUMERIC_MATCH
    )

    engine = VerificationEngine(
        numeric_verifier=verifier,
    )

    result = engine.verify(
        make_claim(ClaimType.NUMERIC),
        "Revenue was 42.8.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.NUMERIC_MATCH
    assert verifier.calls == 1


def test_date_claim_uses_date_verifier() -> None:
    verifier = FakeVerifier(
        VerificationReason.PERIOD_MATCH
    )

    engine = VerificationEngine(
        date_verifier=verifier,
    )

    result = engine.verify(
        make_claim(ClaimType.DATE),
        "The date was 2025-12-31.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert verifier.calls == 1


def test_entity_claim_uses_entity_verifier() -> None:
    verifier = FakeVerifier(
        VerificationReason.ENTITY_MATCH
    )

    engine = VerificationEngine(
        entity_verifier=verifier,
    )

    result = engine.verify(
        make_claim(ClaimType.ENTITY),
        "The entity was Acme.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.ENTITY_MATCH
    assert verifier.calls == 1


def test_period_is_verified_when_claim_has_period() -> None:
    verifier = FakeVerifier(
        VerificationReason.PERIOD_MATCH
    )

    engine = VerificationEngine(
        period_verifier=verifier,
    )

    result = engine.verify(
        make_claim(
            ClaimType.TEXT,
            period="FY2025",
        ),
        "The financial year was 2025.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.PERIOD_MATCH
    assert verifier.calls == 1


def test_unsupported_claim_is_inconclusive() -> None:
    engine = VerificationEngine()

    result = engine.verify(
        make_claim(ClaimType.TEXT),
        "The company performed well.",
    )

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.reason == VerificationReason.UNSUPPORTED_CLAIM
    assert result.confidence == 1.0