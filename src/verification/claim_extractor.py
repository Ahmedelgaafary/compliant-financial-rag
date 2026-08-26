import re
from dataclasses import dataclass

from src.verification.models import Claim, ClaimType


@dataclass(frozen=True)
class ExtractionResult:
    """Result of deterministic claim extraction."""

    claim: Claim | None
    matched_text: str | None = None


class ClaimExtractor:
    """Extract structured financial claims from text."""

    _NUMBER_PATTERN = (
        r"(?P<number>"
        r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"
        r"|\d+(?:\.\d+)?"
        r")"
    )

    _CURRENCY_PATTERN = (
        r"(?P<currency>"
        r"\$|€|£|USD|EUR|GBP"
        r")?"
    )

    _UNIT_PATTERN = (
        r"(?P<unit>"
        r"billion|million|thousand|bn|mn|k"
        r")?"
    )

    _YEAR_PATTERN = r"(?P<year>20\d{2})"

    _NUMERIC_CLAIM_PATTERN = re.compile(
        rf"""
        (?P<subject>
            [A-Za-z][A-Za-z0-9\s\-]{{1,80}}?
        )
        \s+
        (?P<verb>
            was|were|is|are|reached|reported|stood\s+at|
            increased\s+to|decreased\s+to|rose\s+to|
            fell\s+to|totaled|amounted\s+to
        )
        \s+
        {_CURRENCY_PATTERN}
        \s*
        {_NUMBER_PATTERN}
        \s*
        {_UNIT_PATTERN}
        (?:\s+(?P<currency_after>USD|EUR|GBP))?
        (?:\s+in\s+{_YEAR_PATTERN})?
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def extract(
        self,
        text: str,
        claim_id: str,
        source_chunk_id: str | None = None,
    ) -> ExtractionResult:
        """Extract the first supported financial claim."""

        if not text or not text.strip():
            return ExtractionResult(
                claim=None,
                matched_text=None,
            )

        match = self._NUMERIC_CLAIM_PATTERN.search(text)

        if match is None:
            return ExtractionResult(
                claim=None,
                matched_text=None,
            )

        groups = match.groupdict()

        subject = self._clean_subject(groups["subject"])
        number = groups["number"].replace(",", "")

        currency = (
            groups.get("currency")
            or groups.get("currency_after")
        )

        unit = groups.get("unit")

        normalized_unit = self._normalize_unit(
            unit=unit,
            currency=currency,
        )

        period = groups.get("year")

        value = number

        if currency:
            value = f"{currency} {value}"

        claim = Claim(
            claim_id=claim_id,
            claim_type=ClaimType.NUMERIC,
            subject=subject,
            value=value,
            unit=normalized_unit,
            period=period,
            source_chunk_id=source_chunk_id,
        )

        return ExtractionResult(
            claim=claim,
            matched_text=match.group(0),
        )

    @staticmethod
    def _clean_subject(subject: str) -> str:
        """Normalize the extracted claim subject."""

        return " ".join(subject.split()).strip()

    @staticmethod
    def _normalize_unit(
        unit: str | None,
        currency: str | None,
    ) -> str | None:
        """Normalize financial magnitude units."""

        if unit is None:
            return currency

        normalized = unit.lower()

        aliases = {
            "bn": "billion",
            "mn": "million",
            "k": "thousand",
        }

        normalized = aliases.get(
            normalized,
            normalized,
        )

        if currency:
            return f"{currency} {normalized}"

        return normalized