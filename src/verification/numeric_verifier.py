import math
import re

from src.utils.logging import get_logger
from src.verification.models import (
    Claim,
    VerificationResult,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason

logger = get_logger(__name__)


class NumericVerifier:
    """Deterministically verify numeric financial claims."""

    _NUMBER_PATTERN = re.compile(
        r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    )

    _PERCENT_PATTERN = re.compile(
        r"[-+]?(?:\d+(?:\.\d+)?)\s*%"
    )

    _UNIT_MULTIPLIERS = {
        "thousand": 1_000.0,
        "million": 1_000_000.0,
        "billion": 1_000_000_000.0,
        "trillion": 1_000_000_000_000.0,
    }

    def verify(
        self,
        claim: Claim,
        evidence: str,
    ) -> VerificationResult:
        """Verify a numeric claim against evidence text."""

        if claim.claim_type.value != "numeric":
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        claim_value = self._parse_value(
            claim.value,
            claim.unit,
        )

        if claim_value is None:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        evidence_values = self._extract_values(evidence)

        if not evidence_values:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.EVIDENCE_MISSING,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        for evidence_value in evidence_values:
            if math.isclose(
                claim_value,
                evidence_value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status=VerificationStatus.VERIFIED,
                    reason=VerificationReason.NUMERIC_MATCH,
                    confidence=1.0,
                    evidence_chunk_id=claim.source_chunk_id,
                )

        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.REJECTED,
            reason=VerificationReason.NUMERIC_MISMATCH,
            confidence=1.0,
            evidence_chunk_id=claim.source_chunk_id,
        )

    def _extract_values(self, text: str) -> list[float]:
        """Extract normalized numeric values from evidence."""

        values: list[float] = []

        percentage_matches = self._PERCENT_PATTERN.findall(text)

        for match in percentage_matches:
            number_match = self._NUMBER_PATTERN.search(match)

            if number_match is None:
                continue

            try:
                values.append(
                    float(
                        number_match.group(0).replace(",", "")
                    )
                )
            except ValueError:
                continue

        for match in self._NUMBER_PATTERN.finditer(text):
            raw_number = match.group(0)

            end_position = match.end()

            if (
                end_position < len(text)
                and text[end_position] == "%"
            ):
                continue

            try:
                number = float(raw_number.replace(",", ""))
            except ValueError:
                continue

            remaining_text = text[match.end():].lower()

            multiplier = 1.0

            for unit, unit_multiplier in (
                self._UNIT_MULTIPLIERS.items()
            ):
                if remaining_text.lstrip().startswith(unit):
                    multiplier = unit_multiplier
                    break

            values.append(number * multiplier)

        return values

    def _parse_value(
        self,
        value: str,
        unit: str | None,
    ) -> float | None:
        """Normalize a claim value and unit."""

        match = self._NUMBER_PATTERN.search(value)

        if match is None:
            return None

        try:
            number = float(
                match.group(0).replace(",", "")
            )
        except ValueError:
            return None

        if unit == "%":
            return number

        multiplier = 1.0

        if unit:
            normalized_unit = unit.lower()

            for name, unit_multiplier in (
                self._UNIT_MULTIPLIERS.items()
            ):
                if name in normalized_unit:
                    multiplier = unit_multiplier
                    break

        return number * multiplier