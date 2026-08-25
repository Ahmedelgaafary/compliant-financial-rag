from src.verification.citation_verifier import CitationVerifier
from src.verification.contradiction_detector import (
    ContradictionDetector,
)
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
        self.numeric_verifier = (
            numeric_verifier or NumericVerifier()
        )
        self.date_verifier = (
            date_verifier or DateVerifier()
        )
        self.entity_verifier = (
            entity_verifier or EntityVerifier()
        )
        self.period_verifier = (
            period_verifier or PeriodVerifier()
        )
        self.citation_verifier = (
            citation_verifier or CitationVerifier()
        )
        self.contradiction_detector = (
            contradiction_detector or ContradictionDetector()
        )

    def verify(
        self,
        claim: Claim,
        evidence_text: str,
    ) -> VerificationResult:
        """Run deterministic verification checks."""

        contradiction = self.contradiction_detector.verify(
            claim=claim,
            evidence=evidence_text,
        )

        if contradiction.status == VerificationStatus.REJECTED:
            return contradiction

        results: list[VerificationResult] = []

        if claim.claim_type == ClaimType.NUMERIC:
            results.append(
                self.numeric_verifier.verify(
                    claim,
                    evidence_text,
                )
            )

        elif claim.claim_type == ClaimType.DATE:
            results.append(
                self.date_verifier.verify(
                    claim,
                    evidence_text,
                )
            )

        elif claim.claim_type == ClaimType.ENTITY:
            results.append(
                self.entity_verifier.verify(
                    claim,
                    evidence_text,
                )
            )

        elif claim.claim_type == ClaimType.TEXT:
            if not claim.period:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status=VerificationStatus.INCONCLUSIVE,
                    reason=VerificationReason.UNSUPPORTED_CLAIM,
                    confidence=1.0,
                    evidence_chunk_id=claim.source_chunk_id,
                )

        else:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

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
        """Combine individual verification results."""

        if not results:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        rejected = [
            result
            for result in results
            if result.status == VerificationStatus.REJECTED
        ]

        if rejected:
            first_rejection = rejected[0]

            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.REJECTED,
                reason=first_rejection.reason,
                confidence=first_rejection.confidence,
                evidence_chunk_id=first_rejection.evidence_chunk_id,
            )

        inconclusive = [
            result
            for result in results
            if result.status == VerificationStatus.INCONCLUSIVE
        ]

        if inconclusive:
            first_inconclusive = inconclusive[0]

            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=first_inconclusive.reason,
                confidence=first_inconclusive.confidence,
                evidence_chunk_id=first_inconclusive.evidence_chunk_id,
            )

        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.VERIFIED,
            reason=results[0].reason,
            confidence=min(
                result.confidence
                for result in results
            ),
            evidence_chunk_id=claim.source_chunk_id,
        )