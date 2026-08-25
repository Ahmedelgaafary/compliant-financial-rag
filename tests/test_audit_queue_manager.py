import pytest

from src.verification.audit_queue import (
    AuditQueueItem,
    AuditQueueStatus,
)
from src.verification.audit_queue_manager import (
    AuditQueueManager,
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


def _claim(
    claim_id: str = "claim-001",
) -> Claim:
    return Claim(
        claim_id=claim_id,
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


def _item(
    queue_id: str = "audit-001",
) -> AuditQueueItem:
    return AuditQueueItem.create(
        queue_id=queue_id,
        claim=_claim(),
        verification=_verification(),
        risk=_risk(),
        provenance=_provenance(),
        evidence_text=(
            "Total revenue was $42.8 billion in 2025."
        ),
    )


def test_add_and_get_item() -> None:
    manager = AuditQueueManager()
    item = _item()

    manager.add(item)

    result = manager.get("audit-001")

    assert result == item
    assert result.status == AuditQueueStatus.PENDING


def test_duplicate_queue_id_is_rejected() -> None:
    manager = AuditQueueManager()

    manager.add(_item("audit-001"))

    with pytest.raises(ValueError):
        manager.add(_item("audit-001"))


def test_missing_item_raises_key_error() -> None:
    manager = AuditQueueManager()

    with pytest.raises(KeyError):
        manager.get("missing")


def test_pending_returns_only_pending_items() -> None:
    manager = AuditQueueManager()

    manager.add(_item("audit-001"))
    manager.add(_item("audit-002"))

    manager.approve("audit-001")

    pending = manager.pending()

    assert len(pending) == 1
    assert pending[0].queue_id == "audit-002"


def test_approve_item() -> None:
    manager = AuditQueueManager()

    manager.add(_item())

    result = manager.approve("audit-001")

    assert result.status == AuditQueueStatus.APPROVED
    assert manager.get("audit-001").status == (
        AuditQueueStatus.APPROVED
    )


def test_reject_item() -> None:
    manager = AuditQueueManager()

    manager.add(_item())

    result = manager.reject("audit-001")

    assert result.status == AuditQueueStatus.REJECTED
    assert manager.get("audit-001").status == (
        AuditQueueStatus.REJECTED
    )


def test_resolved_item_is_not_pending() -> None:
    manager = AuditQueueManager()

    manager.add(_item())

    manager.reject("audit-001")

    assert manager.pending() == []


def test_resolved_item_cannot_be_approved_again() -> None:
    manager = AuditQueueManager()

    manager.add(_item())
    manager.approve("audit-001")

    with pytest.raises(ValueError):
        manager.approve("audit-001")


def test_resolved_item_cannot_be_rejected_again() -> None:
    manager = AuditQueueManager()

    manager.add(_item())
    manager.reject("audit-001")

    with pytest.raises(ValueError):
        manager.reject("audit-001")


def test_history_preserves_resolved_items() -> None:
    manager = AuditQueueManager()

    manager.add(_item("audit-001"))
    manager.add(_item("audit-002"))

    manager.approve("audit-001")

    history = manager.history()

    assert len(history) == 2
    assert {
        item.queue_id
        for item in history
    } == {"audit-001", "audit-002"}