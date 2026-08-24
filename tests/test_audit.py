# tests/test_audit.py (updated)
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from src.audit.models import AuditRecord, AuditStatus, ReviewDecision
from src.audit.router import AuditRouter, RoutingAction
from src.audit.queue import AuditQueue
from src.audit.decisions import DecisionEngine, ReviewRecommendation
from src.audit.audit_log import AuditLogger
from src.audit.review_service import ReviewService
from src.guardrails.risk_engine import RiskAssessment
from src.verification.models import VerificationResult


# Tests for DecisionEngine
def test_decision_engine_approve():
    engine = DecisionEngine()
    record = AuditRecord(
        audit_id="test",
        timestamp=datetime.now(),
        user_query="test",
        claim="Revenue is $42.8B",
        verification_status="VERIFIED",
        verification_reason="NUMERIC_MATCH",
        risk_level="LOW",
        evidence=[],
        document_id="doc1",
        document_sha256="sha1",
        page_number=1,
        triggers=[],
    )
    result = engine.analyze(record)
    assert result.recommendation == ReviewRecommendation.APPROVE
    assert result.confidence > 0.9


def test_decision_engine_escalate_on_numeric_mismatch():
    engine = DecisionEngine()
    record = AuditRecord(
        audit_id="test",
        timestamp=datetime.now(),
        user_query="test",
        claim="Revenue is $42.8B",
        verification_status="REJECTED",
        verification_reason="NUMERIC_MISMATCH",
        risk_level="HIGH",
        evidence=[],
        document_id="doc1",
        document_sha256="sha1",
        page_number=1,
        triggers=["NUMERIC_MISMATCH"],
    )
    result = engine.analyze(record)
    assert result.recommendation == ReviewRecommendation.ESCALATE


# Tests for AuditLogger
def test_audit_logger_write_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = AuditLogger(log_dir=tmpdir)
        record = AuditRecord(
            audit_id="test123",
            timestamp=datetime.now(),
            user_query="What is revenue?",
            claim="$42.8B",
            verification_status="VERIFIED",
            verification_reason="NUMERIC_MATCH",
            risk_level="LOW",
            evidence=[],
            document_id="doc1",
            document_sha256="sha1",
            page_number=1,
        )
        logger.log(record)
        # Check file exists
        log_file = Path(tmpdir) / "audit_records.json"
        assert log_file.exists()
        with open(log_file, "r") as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["audit_id"] == "test123"


# Tests for ReviewService (integration)
def test_review_service_high_risk_routes_to_human():
    service = ReviewService()
    risk = RiskAssessment(
        risk_score=0.9,
        risk_level="HIGH",
        triggers=["NUMERIC_MISMATCH"],
        recommended_action="HUMAN_REVIEW",
    )
    outcome = service.initiate_review(
        user_query="What is revenue?",
        claim="$45.2B",
        verification_status="REJECTED",
        verification_reason="NUMERIC_MISMATCH",
        risk_assessment=risk,
        verification_results=[],
        evidence=[],
        document_id="doc1",
        document_sha256="sha1",
        page_number=1,
    )
    assert outcome.final_action == "HUMAN_REVIEW"
    assert outcome.audit_record is not None
    assert outcome.routing_decision.action == RoutingAction.HUMAN_REVIEW

    # Check queue has pending
    pending = service.get_pending_reviews()
    assert len(pending) == 1
    assert pending[0].audit_id == outcome.audit_id


def test_review_service_submit_decision():
    service = ReviewService()
    # First create a case
    risk = RiskAssessment(
        risk_score=0.8,
        risk_level="HIGH",
        triggers=["NUMERIC_MISMATCH"],
        recommended_action="HUMAN_REVIEW",
    )
    outcome = service.initiate_review(
        user_query="test",
        claim="test",
        verification_status="REJECTED",
        verification_reason="NUMERIC_MISMATCH",
        risk_assessment=risk,
        verification_results=[],
        evidence=[],
        document_id="doc1",
        document_sha256="sha1",
        page_number=1,
    )
    audit_id = outcome.audit_id

    # Submit decision
    success = service.submit_review_decision(
        audit_id=audit_id,
        decision=ReviewDecision.APPROVED,
        notes="Looks fine",
        reviewer="auditor@example.com",
    )
    assert success is True

    # Check that record is resolved
    record = service.queue.get_by_id(audit_id)
    assert record.status == AuditStatus.RESOLVED
    assert record.review_decision == ReviewDecision.APPROVED