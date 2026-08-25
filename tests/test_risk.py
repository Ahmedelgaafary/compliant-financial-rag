from src.verification.models import (
    VerificationResult,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason
from src.verification.risk import (
    RiskAssessment,
    RiskClassifier,
    RiskLevel,
    RoutingDecision,
)


def _result(
    status: VerificationStatus,
    confidence: float,
) -> VerificationResult:
    return VerificationResult(
        claim_id="claim-001",
        status=status,
        reason=VerificationReason.NUMERIC_MATCH,
        confidence=confidence,
        evidence_chunk_id="chunk-001",
    )


def test_verified_high_confidence_is_low_risk() -> None:
    classifier = RiskClassifier()

    assessment = classifier.classify(
        _result(
            VerificationStatus.VERIFIED,
            1.0,
        )
    )

    assert isinstance(assessment, RiskAssessment)
    assert assessment.level == RiskLevel.LOW
    assert assessment.decision == RoutingDecision.AUTO_APPROVE


def test_verified_lower_confidence_requires_review() -> None:
    classifier = RiskClassifier()

    assessment = classifier.classify(
        _result(
            VerificationStatus.VERIFIED,
            0.95,
        )
    )

    assert assessment.level == RiskLevel.MEDIUM
    assert assessment.decision == RoutingDecision.REVIEW


def test_inconclusive_requires_human_audit() -> None:
    classifier = RiskClassifier()

    assessment = classifier.classify(
        _result(
            VerificationStatus.INCONCLUSIVE,
            0.5,
        )
    )

    assert assessment.level == RiskLevel.HIGH
    assert assessment.decision == RoutingDecision.HUMAN_AUDIT


def test_rejected_requires_human_audit() -> None:
    classifier = RiskClassifier()

    assessment = classifier.classify(
        _result(
            VerificationStatus.REJECTED,
            1.0,
        )
    )

    assert assessment.level == RiskLevel.HIGH
    assert assessment.decision == RoutingDecision.HUMAN_AUDIT


def test_verified_boundary_confidence_is_low_risk() -> None:
    classifier = RiskClassifier()

    assessment = classifier.classify(
        _result(
            VerificationStatus.VERIFIED,
            0.99,
        )
    )

    assert assessment.level == RiskLevel.LOW
    assert assessment.decision == RoutingDecision.AUTO_APPROVE


def test_risk_assessment_contains_reason() -> None:
    classifier = RiskClassifier()

    assessment = classifier.classify(
        _result(
            VerificationStatus.REJECTED,
            1.0,
        )
    )

    assert assessment.reason