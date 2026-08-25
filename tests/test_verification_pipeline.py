from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)
from src.verification.provenance import EvidenceProvenance
from src.verification.reasons import VerificationReason
from src.verification.risk import (
    RiskLevel,
    RoutingDecision,
)
from src.verification.verification_pipeline import (
    VerificationPipeline,
)


class FakeClaimVerifier:
    """Deterministic verifier used to test orchestration."""

    def __init__(
        self,
        status: VerificationStatus,
        confidence: float,
    ) -> None:
        self.status = status
        self.confidence = confidence

    def verify(
        self,
        claim: Claim,
        evidence_text: str,
    ) -> VerificationResult:
        return VerificationResult(
            claim_id=claim.claim_id,
            status=self.status,
            reason=VerificationReason.NUMERIC_MATCH,
            confidence=self.confidence,
            evidence_chunk_id=claim.source_chunk_id,
        )


def _claim() -> Claim:
    return Claim(
        claim_id="claim-001",
        claim_type=ClaimType.NUMERIC,
        subject="revenue",
        value="42.8",
        unit="USD billion",
        period="2025",
        source_chunk_id="chunk-047",
    )


def _provenance() -> EvidenceProvenance:
    return EvidenceProvenance(
        document_id="annual-report-2025",
        document_hash="a" * 64,
        chunk_id="chunk-047",
        page_number=47,
        section="Consolidated Statements",
    )


def test_verified_claim_is_automatically_approved() -> None:
    verifier = FakeClaimVerifier(
        status=VerificationStatus.VERIFIED,
        confidence=1.0,
    )

    pipeline = VerificationPipeline(
        claim_verifier=verifier,
    )

    result = pipeline.process(
        claim=_claim(),
        evidence_text="Revenue was $42.8 billion.",
        provenance=_provenance(),
    )

    assert result.verification.status == (
        VerificationStatus.VERIFIED
    )
    assert result.risk.level == RiskLevel.LOW
    assert result.risk.decision == (
        RoutingDecision.AUTO_APPROVE
    )
    assert result.audit_queue_item is None


def test_inconclusive_claim_is_sent_to_human_audit() -> None:
    verifier = FakeClaimVerifier(
        status=VerificationStatus.INCONCLUSIVE,
        confidence=0.5,
    )

    pipeline = VerificationPipeline(
        claim_verifier=verifier,
    )

    result = pipeline.process(
        claim=_claim(),
        evidence_text="Revenue information is unclear.",
        provenance=_provenance(),
        queue_id="audit-001",
    )

    assert result.risk.level == RiskLevel.HIGH
    assert result.risk.decision == (
        RoutingDecision.HUMAN_AUDIT
    )
    assert result.audit_queue_item is not None
    assert result.audit_queue_item.queue_id == "audit-001"

    assert len(pipeline.audit_queue.pending()) == 1


def test_rejected_claim_is_sent_to_human_audit() -> None:
    verifier = FakeClaimVerifier(
        status=VerificationStatus.REJECTED,
        confidence=1.0,
    )

    pipeline = VerificationPipeline(
        claim_verifier=verifier,
    )

    result = pipeline.process(
        claim=_claim(),
        evidence_text="Revenue was $30 billion.",
        provenance=_provenance(),
        queue_id="audit-002",
    )

    assert result.verification.status == (
        VerificationStatus.REJECTED
    )
    assert result.risk.level == RiskLevel.HIGH
    assert result.audit_queue_item is not None


def test_high_risk_requires_queue_id() -> None:
    verifier = FakeClaimVerifier(
        status=VerificationStatus.REJECTED,
        confidence=1.0,
    )

    pipeline = VerificationPipeline(
        claim_verifier=verifier,
    )

    try:
        pipeline.process(
            claim=_claim(),
            evidence_text="Revenue was $30 billion.",
            provenance=_provenance(),
        )
    except ValueError as exc:
        assert "queue_id" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when queue_id is missing."
        )