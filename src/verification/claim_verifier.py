from src.verification.models import (
    Claim,
    VerificationResult,
    VerificationStatus,
)
from src.verification.numeric_verifier import NumericVerifier
from src.verification.period_verifier import PeriodVerifier
from src.verification.reasons import VerificationReason


class ClaimVerifier:
    """Combine deterministic verification checks."""

    def __init__(
        self,
        numeric_verifier: NumericVerifier | None = None,
        period_verifier: PeriodVerifier | None = None,
    ) -> None:
        self.numeric_verifier = (
            numeric_verifier or NumericVerifier()
        )
        self.period_verifier = (
            period_verifier or PeriodVerifier()
        )

    def verify(
        self,
        claim: Claim,
        evidence: str,
    ) -> VerificationResult:
        """Verify all applicable properties of a claim."""

        results: list[VerificationResult] = []

        if claim.claim_type.value == "numeric":
            results.append(
                self.numeric_verifier.verify(
                    claim,
                    evidence,
                )
            )

        if claim.period:
            results.append(
                self.period_verifier.verify(
                    claim,
                    evidence,
                )
            )

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
                evidence_chunk_id=(
                    first_rejection.evidence_chunk_id
                ),
            )

        inconclusive = [
            result
            for result in results
            if result.status
            == VerificationStatus.INCONCLUSIVE
        ]

        if inconclusive:
            first_inconclusive = inconclusive[0]

            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=first_inconclusive.reason,
                confidence=first_inconclusive.confidence,
                evidence_chunk_id=(
                    first_inconclusive.evidence_chunk_id
                ),
            )

        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.VERIFIED,
            reason=VerificationReason.NUMERIC_MATCH,
            confidence=1.0,
            evidence_chunk_id=claim.source_chunk_id,
        )