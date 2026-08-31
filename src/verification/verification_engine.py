"""Deterministic verification engine."""

from __future__ import annotations

from src.verification.citation_verifier import CitationVerifier
from src.verification.contradiction_detector import ContradictionDetector
from src.verification.date_verifier import DateVerifier
from src.verification.entity_verifier import EntityVerifier
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)
from src.verification.numeric_verifier import NumericVerifier
from src.verification.period_verifier import PeriodVerifier
from src.verification.reasons import VerificationReason


class VerificationEngine:
    """Coordinate deterministic verification checks."""

    def __init__(
        self,
        numeric_verifier: NumericVerifier | None = None,
        date_verifier: DateVerifier | None = None,
        entity_verifier: EntityVerifier | None = None,
        period_verifier: PeriodVerifier | None = None,
        citation_verifier: CitationVerifier | None = None,
        contradiction_detector: ContradictionDetector | None = None,
    ) -> None:
        self.numeric_verifier = numeric_verifier or NumericVerifier()
        self.date_verifier = date_verifier or DateVerifier()
        self.entity_verifier = entity_verifier or EntityVerifier()
        self.period_verifier = period_verifier or PeriodVerifier()
        self.citation_verifier = (
            citation_verifier or CitationVerifier()
        )
        self.contradiction_detector = (
            contradiction_detector or ContradictionDetector()
        )

    def verify(
        self,
        claim: Claim,
        evidence_text: str | None,
    ) -> VerificationResult:
        """Verify one claim against its scoped evidence."""
        if not evidence_text or not evidence_text.strip():
            return self._result(
                claim,
                VerificationStatus.INCONCLUSIVE,
                VerificationReason.EVIDENCE_MISSING,
                1.0,
            )

        contradiction = self.contradiction_detector.verify(
            claim=claim,
            evidence=evidence_text,
        )

        if contradiction.status == VerificationStatus.REJECTED:
            return contradiction

        if claim.claim_type == ClaimType.NUMERIC:
            result = self.numeric_verifier.verify(
                claim,
                evidence_text,
            )

        elif claim.claim_type == ClaimType.DATE:
            result = self.date_verifier.verify(
                claim,
                evidence_text,
            )

        elif claim.claim_type == ClaimType.ENTITY:
            result = self.entity_verifier.verify(
                claim,
                evidence_text,
            )

        elif claim.claim_type == ClaimType.TEXT:
            if claim.period:
                result = self.period_verifier.verify(
                    claim,
                    evidence_text,
                )
                return self._combine_results(
                    claim,
                    [result],
                )

            return self._result(
                claim,
                VerificationStatus.INCONCLUSIVE,
                VerificationReason.UNSUPPORTED_CLAIM,
                1.0,
            )

        else:
            return self._result(
                claim,
                VerificationStatus.INCONCLUSIVE,
                VerificationReason.UNSUPPORTED_CLAIM,
                1.0,
            )

        results = [result]

        if claim.period:
            results.append(
                self.period_verifier.verify(
                    claim,
                    evidence_text,
                )
            )

        return self._combine_results(
            claim,
            results,
        )

    def _combine_results(
        self,
        claim: Claim,
        results: list[VerificationResult],
    ) -> VerificationResult:
        """Combine deterministic verification checks."""
        if not results:
            return self._result(
                claim,
                VerificationStatus.INCONCLUSIVE,
                VerificationReason.UNSUPPORTED_CLAIM,
                1.0,
            )

        rejected = [
            result
            for result in results
            if result.status == VerificationStatus.REJECTED
        ]

        if rejected:
            first = rejected[0]
            return self._result(
                claim,
                VerificationStatus.REJECTED,
                first.reason,
                first.confidence,
                evidence_chunk_id=first.evidence_chunk_id,
                normalized_value=first.normalized_value,
            )

        inconclusive = [
            result
            for result in results
            if result.status == VerificationStatus.INCONCLUSIVE
        ]

        if inconclusive:
            first = inconclusive[0]
            return self._result(
                claim,
                VerificationStatus.INCONCLUSIVE,
                first.reason,
                first.confidence,
                evidence_chunk_id=first.evidence_chunk_id,
                normalized_value=first.normalized_value,
            )

        return self._result(
            claim,
            VerificationStatus.VERIFIED,
            results[0].reason,
            min(result.confidence for result in results),
            evidence_chunk_id=claim.source_chunk_id,
            normalized_value=next(
                (
                    result.normalized_value
                    for result in results
                    if result.normalized_value is not None
                ),
                None,
            ),
        )

    @staticmethod
    def _result(
        claim: Claim,
        status: VerificationStatus,
        reason: str,
        confidence: float,
        evidence_chunk_id: str | None = None,
        normalized_value: float | None = None,
    ) -> VerificationResult:
        """Build a complete verification result."""
        return VerificationResult(
            claim_id=claim.claim_id,
            status=status,
            reason=reason,
            confidence=confidence,
            evidence_chunk_id=(
                evidence_chunk_id
                if evidence_chunk_id is not None
                else claim.source_chunk_id
            ),
            claim_type=claim.claim_type,
            company_name=claim.company_name,
            question_id=claim.question_id,
            normalized_value=normalized_value,
        )