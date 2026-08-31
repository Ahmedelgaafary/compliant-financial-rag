"""Deterministic numeric verification for financial claims."""

from __future__ import annotations

import math
import re

from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason


class NumericVerifier:
    """Deterministically verify numeric financial claims."""

    _NUMBER = (
        r"[-+]?"
        r"(?:"
        r"\d{1,3}(?:,\d{3})+"
        r"|"
        r"\d+(?:\.\d+)?"
        r")"
    )

    _PERCENT_PATTERN = re.compile(
        rf"(?P<number>{_NUMBER})\s*%",
        re.IGNORECASE,
    )

    _FINANCIAL_SUFFIX_PATTERN = re.compile(
        rf"(?P<number>{_NUMBER})"
        r"\s*"
        r"(?P<suffix>"
        r"bn|mn|b|m|k|"
        r"billion|million|thousand|trillion"
        r")\b",
        re.IGNORECASE,
    )

    _PARENTHESIZED_NUMBER_PATTERN = re.compile(
        rf"\(\s*(?P<number>{_NUMBER})\s*\)"
    )

    _PLAIN_NUMBER_PATTERN = re.compile(
        _NUMBER
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

    _CONTEXT_UNIT_PATTERN = re.compile(
        r"\(\s*in\s+"
        r"(thousands?|millions?|billions?|trillions?)"
        r"(?:\s+of\s+\w+)?"
        r"\s*\)",
        re.IGNORECASE,
    )

    def verify(
        self,
        claim: Claim,
        evidence: str | None,
    ) -> VerificationResult:
        """Verify a numeric claim against evidence."""

        base_kwargs = {
            "claim_id": claim.claim_id,
            "claim_type": ClaimType.NUMERIC,
            "company_name": claim.company_name,
            "question_id": claim.question_id,
            "evidence_chunk_id": claim.source_chunk_id,
        }

        if claim.claim_type != ClaimType.NUMERIC:
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=1.0,
                **base_kwargs,
            )

        if not evidence or not evidence.strip():
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.EVIDENCE_MISSING,
                confidence=1.0,
                **base_kwargs,
            )

        claim_value = self._parse_value(
            claim.value,
            claim.unit,
        )

        if claim_value is None:
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=1.0,
                **base_kwargs,
            )

        evidence_values = self._extract_values(
            evidence
        )

        if not evidence_values:
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.EVIDENCE_MISSING,
                confidence=1.0,
                **base_kwargs,
            )

        for evidence_value in evidence_values:
            if math.isclose(
                claim_value,
                evidence_value,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    reason=VerificationReason.NUMERIC_MATCH,
                    confidence=1.0,
                    normalized_value=claim_value,
                    **base_kwargs,
                )

        return VerificationResult(
            status=VerificationStatus.REJECTED,
            reason=VerificationReason.NUMERIC_MISMATCH,
            confidence=1.0,
            normalized_value=claim_value,
            **base_kwargs,
        )

    def _extract_values(
        self,
        text: str,
    ) -> list[float]:
        """
        Extract normalized numeric values.

        Supports:

        - 42
        - 42.8
        - $42.8B
        - $42.8 billion
        - (8,066)
        - 42.8%
        - tables explicitly marked "(in millions)"
        """

        values: list[float] = []
        consumed_spans: list[tuple[int, int]] = []

        # ---------------------------------------------------------
        # Explicit context unit.
        # ---------------------------------------------------------
        context_multiplier = self._context_multiplier(text)

        # ---------------------------------------------------------
        # Percentages.
        # ---------------------------------------------------------
        for match in self._PERCENT_PATTERN.finditer(text):
            number = self._safe_float(
                match.group("number")
            )

            if number is not None:
                values.append(number)

            consumed_spans.append(
                (
                    match.start(),
                    match.end(),
                )
            )

        # ---------------------------------------------------------
        # Explicit financial suffixes.
        # ---------------------------------------------------------
        for match in self._FINANCIAL_SUFFIX_PATTERN.finditer(text):
            number = self._safe_float(
                match.group("number")
            )

            if number is None:
                continue

            suffix = match.group("suffix").casefold()

            multiplier = self._UNIT_MULTIPLIERS.get(
                suffix,
                1.0,
            )

            values.append(
                number * multiplier
            )

            consumed_spans.append(
                (
                    match.start(),
                    match.end(),
                )
            )

        # ---------------------------------------------------------
        # Parenthesized negative values.
        # ---------------------------------------------------------
        for match in self._PARENTHESIZED_NUMBER_PATTERN.finditer(
            text
        ):
            if self._span_consumed(
                match.start(),
                consumed_spans,
            ):
                continue

            number = self._safe_float(
                match.group("number")
            )

            if number is None:
                continue

            values.append(
                -number * context_multiplier
            )

            consumed_spans.append(
                (
                    match.start(),
                    match.end(),
                )
            )

        # ---------------------------------------------------------
        # Plain numeric values.
        # ---------------------------------------------------------
        for match in self._PLAIN_NUMBER_PATTERN.finditer(text):
            if self._span_consumed(
                match.start(),
                consumed_spans,
            ):
                continue

            # Skip percentages.
            if (
                match.end() < len(text)
                and text[match.end()] == "%"
            ):
                continue

            number = self._safe_float(
                match.group(0)
            )

            if number is None:
                continue

            values.append(
                number * context_multiplier
            )

        return values

    def _parse_value(
        self,
        value: str,
        unit: str | None,
    ) -> float | None:
        """Normalize a claim value and optional unit."""

        if not value or not value.strip():
            return None

        text = value.strip()

        # Parenthesized negative claim, e.g. (8.2M).
        parenthesized = re.search(
            r"\(\s*(?P<body>[^)]+)\)",
            text,
        )

        sign = 1.0

        if parenthesized:
            text = parenthesized.group("body")
            sign = -1.0

        suffix_match = self._FINANCIAL_SUFFIX_PATTERN.search(
            text
        )

        if suffix_match:
            number = self._safe_float(
                suffix_match.group("number")
            )

            if number is None:
                return None

            suffix = suffix_match.group("suffix").casefold()

            return (
                sign
                * number
                * self._UNIT_MULTIPLIERS.get(
                    suffix,
                    1.0,
                )
            )

        number_match = self._PLAIN_NUMBER_PATTERN.search(
            text
        )

        if number_match is None:
            return None

        number = self._safe_float(
            number_match.group(0)
        )

        if number is None:
            return None

        if unit:
            normalized_unit = unit.casefold().strip()

            if normalized_unit in {"%", "percent", "percentage"}:
                return sign * number

            for name, multiplier in self._UNIT_MULTIPLIERS.items():
                if name in normalized_unit:
                    return sign * number * multiplier

        return sign * number

    def _context_multiplier(
        self,
        text: str,
    ) -> float:
        """Return a table-level magnitude multiplier."""

        match = self._CONTEXT_UNIT_PATTERN.search(text)

        if not match:
            return 1.0

        unit = match.group(1).casefold()

        if unit.endswith("s"):
            unit = unit[:-1]

        return self._UNIT_MULTIPLIERS.get(
            unit,
            1.0,
        )

    @staticmethod
    def _safe_float(value: str) -> float | None:
        try:
            return float(
                value.replace(",", "")
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _span_consumed(
        position: int,
        spans: list[tuple[int, int]],
    ) -> bool:
        return any(
            start <= position < end
            for start, end in spans
        )