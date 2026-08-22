import re

from src.verification.models import (
    Claim,
    VerificationResult,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason


class PeriodVerifier:
    """Deterministically verify financial reporting periods."""

    _YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

    def verify(
        self,
        claim: Claim,
        evidence: str,
    ) -> VerificationResult:
        """Verify the claim period against evidence."""

        if not claim.period:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        evidence_years = self._extract_years(evidence)

        if not evidence_years:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.EVIDENCE_MISSING,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        claim_years = self._extract_years(claim.period)

        if not claim_years:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        if claim_years.intersection(evidence_years):
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.VERIFIED,
                reason=VerificationReason.PERIOD_MATCH,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.REJECTED,
            reason=VerificationReason.PERIOD_MISMATCH,
            confidence=1.0,
            evidence_chunk_id=claim.source_chunk_id,
        )

    def _extract_years(self, text: str) -> set[str]:
        """Extract four-digit years from text."""

        return set(self._YEAR_PATTERN.findall(text))