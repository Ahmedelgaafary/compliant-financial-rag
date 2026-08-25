from src.verification.contradiction_detector import (
    ContradictionDetector,
)
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)
from src.verification.numeric_verifier import NumericVerifier
from src.verification.reasons import VerificationReason
from src.verification.verification_engine import (
    VerificationEngine,
)


def make_claim(
    value: str = "42.8",
    claim_type: ClaimType = ClaimType.NUMERIC,
) -> Claim:
    return Claim(
        claim_id="claim-001",
        claim_type=claim_type,
        subject="revenue",
        value=value,
        unit="USD billion",
        period="2025",
        source_chunk_id="chunk-001",
    )


def test_explicit_contradiction_rejects_claim() -> None:
    engine = VerificationEngine()

    result = engine.verify(
        make_claim(),
        "Revenue was not 42.8 billion in 2025.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.EVIDENCE_CONTRADICTS
    assert result.confidence == 1.0


def test_supported_claim_is_not_rejected_by_contradiction_check() -> None:
    engine = VerificationEngine()

    result = engine.verify(
        make_claim(),
        "Revenue was 42.8 billion in 2025.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert result.reason == VerificationReason.NUMERIC_MATCH


def test_contradiction_takes_priority_over_numeric_match() -> None:
    engine = VerificationEngine()

    result = engine.verify(
        make_claim(),
        "Revenue was 42.8 billion in 2025, but the statement that "
        "revenue was 42.8 billion is false.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.EVIDENCE_CONTRADICTS


class FakeContradictionDetector:
    def __init__(
        self,
        status: VerificationStatus,
        reason: VerificationReason,
    ) -> None:
        self.status = status
        self.reason = reason
        self.calls = 0

    def verify(
        self,
        claim: Claim,
        evidence: str | None,
    ) -> VerificationResult:
        self.calls += 1

        return VerificationResult(
            claim_id=claim.claim_id,
            status=self.status,
            reason=self.reason,
            confidence=1.0,
            evidence_chunk_id=claim.source_chunk_id,
        )


def test_contradiction_detector_is_called() -> None:
    detector = FakeContradictionDetector(
        status=VerificationStatus.INCONCLUSIVE,
        reason=VerificationReason.UNSUPPORTED_CLAIM,
    )

    engine = VerificationEngine(
        contradiction_detector=detector,
    )

    result = engine.verify(
        make_claim(),
        "Revenue was 42.8 billion in 2025.",
    )

    assert result.status == VerificationStatus.VERIFIED
    assert detector.calls == 1


def test_injected_contradiction_detector_can_reject() -> None:
    detector = FakeContradictionDetector(
        status=VerificationStatus.REJECTED,
        reason=VerificationReason.EVIDENCE_CONTRADICTS,
    )

    engine = VerificationEngine(
        contradiction_detector=detector,
    )

    result = engine.verify(
        make_claim(),
        "Revenue was 42.8 billion in 2025.",
    )

    assert result.status == VerificationStatus.REJECTED
    assert result.reason == VerificationReason.EVIDENCE_CONTRADICTS
    assert detector.calls == 1