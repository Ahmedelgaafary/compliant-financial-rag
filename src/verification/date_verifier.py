import re
from datetime import date, datetime

from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason


class DateVerifier:
    """Deterministically verify specific dates against evidence."""

    _DATE_PATTERNS = (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    )

    _DATE_REGEX = re.compile(
        r"""
        \b
        (?:
            \d{4}-\d{2}-\d{2}
            |
            \d{1,2}/\d{1,2}/\d{4}
            |
            (?:January|February|March|April|May|June|July|
               August|September|October|November|December)
            \s+\d{1,2},\s+\d{4}
            |
            \d{1,2}\s+
            (?:January|February|March|April|May|June|July|
               August|September|October|November|December)
            \s+\d{4}
        )
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def verify(
        self,
        claim: Claim,
        evidence: str | None,
    ) -> VerificationResult:
        """Verify a date claim against evidence."""

        if claim.claim_type != ClaimType.DATE:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        if not claim.value.strip():
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        if not evidence or not evidence.strip():
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.EVIDENCE_MISSING,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        claim_date = self._parse_date(claim.value)

        if claim_date is None:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        evidence_dates = self._extract_dates(evidence)

        if not evidence_dates:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.EVIDENCE_MISSING,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        if claim_date in evidence_dates:
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

    def _extract_dates(self, text: str) -> set[date]:
        """Extract and normalize dates from evidence."""

        dates: set[date] = set()

        for match in self._DATE_REGEX.finditer(text):
            parsed = self._parse_date(match.group(0))

            if parsed is not None:
                dates.add(parsed)

        return dates

    def _parse_date(self, value: str) -> date | None:
        """Parse a supported date representation."""

        normalized = " ".join(value.strip().split())

        for pattern in self._DATE_PATTERNS:
            try:
                return datetime.strptime(
                    normalized,
                    pattern,
                ).date()
            except ValueError:
                continue

        return None