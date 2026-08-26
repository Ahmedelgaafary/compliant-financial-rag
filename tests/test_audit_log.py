from datetime import datetime

import pytest

from src.audit.audit_log import AuditLogger
from src.audit.models import AuditRecord, AuditStatus, ReviewDecision


def make_record(audit_id="AUD-1", **overrides):
    defaults = {
        "audit_id": audit_id,
        "timestamp": datetime.now(),
        "user_query": "What is revenue?",
        "claim": "Revenue was $42.8B",
        "claim_id": "claim-1",
        "verification_status": "VERIFIED",
        "verification_reason": "numeric_match",
        "risk_level": "LOW",
        "evidence": [{"text": "Revenue ...", "page": 3}],
        "provenance": [{"document_id": "doc1", "chunk_id": "chunk1"}],
        "document_id": "doc1",
        "document_sha256": "sha1",
        "page_number": 3,
        "risk_assessment": "low risk",
        "created_at": datetime.now(),
        "reviewer": None,
        "review_decision": None,
        "review_notes": None,
        "review_timestamp": None,
        "status": AuditStatus.PENDING,
        "confidence_score": 0.9,
        "risk_score": 0.1,
        "triggers": [],
        "verification_results": [],
    }
    defaults.update(overrides)
    return AuditRecord(**defaults)


def test_append_only_logging(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    logger.log(make_record(audit_id="AUD-1"))
    logger.log(make_record(audit_id="AUD-2"))
    assert len(logger.get_all()) == 2


def test_duplicate_id_rejected(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    logger.log(make_record(audit_id="AUD-1"))
    with pytest.raises(ValueError):
        logger.log(make_record(audit_id="AUD-1"))


def test_required_fields_preserved(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    rec = make_record(
        audit_id="AUD-1",
        claim_id="claim-42",
        verification_status="REJECTED",
        verification_reason="numeric_mismatch",
        risk_level="HIGH",
        evidence=[{"text": "wrong", "page": 5}],
        provenance=[{"document_id": "docX", "chunk_id": "chunkY"}],
        reviewer="alice",
        review_decision=ReviewDecision.REJECTED,
        review_notes="Not supported",
    )
    logger.log(rec)
    fetched = logger.get_by_audit_id("AUD-1")
    assert fetched.claim_id == "claim-42"
    assert fetched.provenance == [{"document_id": "docX", "chunk_id": "chunkY"}]
    assert fetched.reviewer == "alice"
    assert fetched.review_decision == ReviewDecision.REJECTED


def test_immutable_after_logging(tmp_path):
    logger = AuditLogger(log_dir=str(tmp_path))
    rec = make_record(audit_id="AUD-1", user_query="Original")
    logger.log(rec)
    rec.user_query = "Modified"
    fetched = logger.get_by_audit_id("AUD-1")
    assert fetched.user_query == "Original"