from datetime import timezone

from src.verification.audit_records import AuditRecord
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)
from src.verification.provenance import EvidenceProvenance
from src.verification.reasons import VerificationReason


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
        status=VerificationStatus.VERIFIED,
        reason=VerificationReason.NUMERIC_MATCH,
        confidence=1.0,
        evidence_chunk_id="chunk-047",
    )


def _provenance() -> EvidenceProvenance:
    return EvidenceProvenance(
        document_id="annual-report-2025",
        document_hash="a" * 64,
        chunk_id="chunk-047",
        page_number=47,
        section="Consolidated Statements",
    )


def test_create_audit_record() -> None:
    record = AuditRecord.create(
        claim=_claim(),
        verification=_verification(),
        provenance=_provenance(),
        evidence_text=(
            "Total revenue was $42.8 billion in 2025."
        ),
    )

    assert record.claim.claim_id == "claim-001"
    assert record.verification.status == (
        VerificationStatus.VERIFIED
    )
    assert record.provenance.chunk_id == "chunk-047"
    assert record.evidence_text
    assert record.created_at.tzinfo == timezone.utc


def test_audit_record_is_auditable() -> None:
    record = AuditRecord.create(
        claim=_claim(),
        verification=_verification(),
        provenance=_provenance(),
        evidence_text=(
            "Total revenue was $42.8 billion in 2025."
        ),
    )

    assert record.is_auditable()


def test_audit_record_is_verified() -> None:
    record = AuditRecord.create(
        claim=_claim(),
        verification=_verification(),
        provenance=_provenance(),
        evidence_text=(
            "Total revenue was $42.8 billion in 2025."
        ),
    )

    assert record.is_verified()


def test_empty_evidence_is_not_auditable() -> None:
    record = AuditRecord.create(
        claim=_claim(),
        verification=_verification(),
        provenance=_provenance(),
        evidence_text="",
    )

    assert not record.is_auditable()


def test_mismatched_chunk_is_not_auditable() -> None:
    verification = VerificationResult(
        claim_id="claim-001",
        status=VerificationStatus.VERIFIED,
        reason=VerificationReason.NUMERIC_MATCH,
        confidence=1.0,
        evidence_chunk_id="different-chunk",
    )

    record = AuditRecord.create(
        claim=_claim(),
        verification=verification,
        provenance=_provenance(),
        evidence_text=(
            "Total revenue was $42.8 billion in 2025."
        ),
    )

    assert not record.is_auditable()


def test_rejected_claim_is_not_verified() -> None:
    verification = VerificationResult(
        claim_id="claim-001",
        status=VerificationStatus.REJECTED,
        reason=VerificationReason.NUMERIC_MISMATCH,
        confidence=1.0,
        evidence_chunk_id="chunk-047",
    )

    record = AuditRecord.create(
        claim=_claim(),
        verification=verification,
        provenance=_provenance(),
        evidence_text=(
            "Total revenue was $42.8 billion in 2025."
        ),
    )

    assert record.is_auditable()
    assert not record.is_verified()