
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

    _FINANCIAL_SUFFIX_PATTERN = re.compile(
        r"(?P<number>"
        r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
        r")"
        r"\s*"
        r"(?P<suffix>"
        r"bn|mn|b|m|k|"
        r"billion|million|thousand|trillion"
        r")\b",
        re.IGNORECASE,
    )

    _UNIT_MULTIPLIERS = {
        "thousand": 1_000.0,
        "million": 1_000_000.0,
        "billion": 1_000_000_000.0,
        "trillion": 1_000_000_000_000.0,
        "k": 1_000.0,
        "mn": 1_000_000.0,
        "m": 1_000_000.0,
        "bn": 1_000_000_000.0,
        "b": 1_000_000_000.0,
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

        if not evidence or not evidence.strip():
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.EVIDENCE_MISSING,
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
        """
        Extract normalized numeric values from evidence.

        Supports:
            42
            42.8
            42.8B
            42.8 billion
            42.8M
            42.8 million
            42.8K
            42.8 thousand
            42.8bn
            42.8mn

        Percentages are returned as their numeric percentage value.
        """

        values: list[float] = []

        # ---------------------------------------------------------
        # 1. Percentages
        # ---------------------------------------------------------
        percentage_matches = self._PERCENT_PATTERN.finditer(text)

        for match in percentage_matches:
            number_match = self._NUMBER_PATTERN.search(
                match.group(0)
            )

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

        # ---------------------------------------------------------
        # 2. Financial values with explicit magnitude suffixes
        # ---------------------------------------------------------
        suffix_spans: list[tuple[int, int]] = []

        for match in self._FINANCIAL_SUFFIX_PATTERN.finditer(text):
            number = match.group("number")
            suffix = match.group("suffix").lower()

            try:
                numeric_value = float(
                    number.replace(",", "")
                )
            except ValueError:
                continue

            multiplier = self._UNIT_MULTIPLIERS.get(
                suffix,
                1.0,
            )

            values.append(
                numeric_value * multiplier
            )

            suffix_spans.append(
                (
                    match.start(),
                    match.end(),
                )
            )

        # ---------------------------------------------------------
        # 3. Plain numeric values
        #
        # Skip numbers already consumed by the financial-suffix
        # parser so that 42.8B is not also interpreted as 42.8.
        # ---------------------------------------------------------
        for match in self._NUMBER_PATTERN.finditer(text):
            if any(
                start <= match.start() < end
                for start, end in suffix_spans
            ):
                continue

            raw_number = match.group(0)

            # Skip percentages.
            if (
                match.end() < len(text)
                and text[match.end()] == "%"
            ):
                continue

            try:
                values.append(
                    float(
                        raw_number.replace(",", "")
                    )
                )
            except ValueError:
                continue

        return values

    def _parse_value(
        self,
        value: str,
        unit: str | None,
    ) -> float | None:
        """Normalize a claim value and unit."""

        if not value or not value.strip():
            return None

        # First support values that already contain a suffix,
        # e.g. "$42.8B".
        suffix_match = self._FINANCIAL_SUFFIX_PATTERN.search(
            value
        )

        if suffix_match:
            try:
                number = float(
                    suffix_match.group("number").replace(",", "")
                )
            except ValueError:
                return None

            suffix = suffix_match.group("suffix").lower()

            multiplier = self._UNIT_MULTIPLIERS.get(
                suffix,
                1.0,
            )

            return number * multiplier

        # Otherwise parse the numeric component.
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
            normalized_unit = unit.lower().strip()

            for name, unit_multiplier in (
                self._UNIT_MULTIPLIERS.items()
            ):
                if name in normalized_unit:
                    multiplier = unit_multiplier
                    break

        return number * multiplier

