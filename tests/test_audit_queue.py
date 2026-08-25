from datetime import timezone

import pytest

from src.verification.audit_queue import (
    AuditQueueItem,
    AuditQueueStatus,
)
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)
from src.verification.provenance import EvidenceProvenance
from src.verification.reasons import VerificationReason
from src.verification.risk import (
    RiskAssessment,
    RiskLevel,
    RoutingDecision,
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


def _verification() -> VerificationResult:
    return VerificationResult(
        claim_id="claim-001",
        status=VerificationStatus.REJECTED,
        reason=VerificationReason.NUMERIC_MISMATCH,
        confidence=1.0,
        evidence_chunk_id="chunk-047",
    )


def _risk() -> RiskAssessment:
    return RiskAssessment(
        level=RiskLevel.HIGH,
        decision=RoutingDecision.HUMAN_AUDIT,
        reason="Evidence contradicts the claim.",
    )


def _provenance() -> EvidenceProvenance:
    return EvidenceProvenance(
        document_id="annual-report-2025",
        document_hash="a" * 64,
        chunk_id="chunk-047",
        page_number=47,
        section="Consolidated Statements",
    )


def _create_item() -> AuditQueueItem:
    return AuditQueueItem.create(
        queue_id="audit-001",
        claim=_claim(),
        verification=_verification(),
        risk=_risk(),
        provenance=_provenance(),
        evidence_text=(
            "Total revenue was $42.8 billion in 2025."
        ),
    )


def test_create_pending_audit_item() -> None:
    item = _create_item()

    assert item.queue_id == "audit-001"
    assert item.status == AuditQueueStatus.PENDING
    assert item.risk.level == RiskLevel.HIGH
    assert item.risk.decision == RoutingDecision.HUMAN_AUDIT
    assert item.created_at.tzinfo == timezone.utc


def test_approve_audit_item() -> None:
    item = _create_item()

    approved = item.approve()

    assert item.status == AuditQueueStatus.PENDING
    assert approved.status == AuditQueueStatus.APPROVED
    assert approved.queue_id == item.queue_id
    assert approved.created_at == item.created_at


def test_reject_audit_item() -> None:
    item = _create_item()

    rejected = item.reject()

    assert item.status == AuditQueueStatus.PENDING
    assert rejected.status == AuditQueueStatus.REJECTED
    assert rejected.queue_id == item.queue_id


def test_medium_risk_cannot_enter_human_queue() -> None:
    risk = RiskAssessment(
        level=RiskLevel.MEDIUM,
        decision=RoutingDecision.REVIEW,
        reason="Additional review required.",
    )

    with pytest.raises(ValueError):
        AuditQueueItem.create(
            queue_id="audit-002",
            claim=_claim(),
            verification=_verification(),
            risk=risk,
            provenance=_provenance(),
            evidence_text="Evidence.",
        )


def test_low_risk_cannot_enter_human_queue() -> None:
    risk = RiskAssessment(
        level=RiskLevel.LOW,
        decision=RoutingDecision.AUTO_APPROVE,
        reason="Verified with high confidence.",
    )

    with pytest.raises(ValueError):
        AuditQueueItem.create(
            queue_id="audit-003",
            claim=_claim(),
            verification=_verification(),
            risk=risk,
            provenance=_provenance(),
            evidence_text="Evidence.",
        )


def test_non_human_routing_cannot_enter_queue() -> None:
    risk = RiskAssessment(
        level=RiskLevel.HIGH,
        decision=RoutingDecision.REVIEW,
        reason="Requires additional review.",
    )

    with pytest.raises(ValueError):
        AuditQueueItem.create(
            queue_id="audit-004",
            claim=_claim(),
            verification=_verification(),
            risk=risk,
            provenance=_provenance(),
            evidence_text="Evidence.",
        )