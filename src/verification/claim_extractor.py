import re
from dataclasses import dataclass

from src.verification.models import Claim, ClaimType


@dataclass(frozen=True)
class ExtractionResult:
    """Result of deterministic claim extraction."""

    claim: Claim | None
    matched_text: str | None = None


class ClaimExtractor:
    """
    Extract structured financial claims from text.

    The extractor is intentionally deterministic and limited to numeric
    financial claims. It supports common natural-language forms such as:

        Revenue was $42.8 billion in 2025.
        Revenue was $42.8B in 2025.
        The company reported revenue of $42.8B in 2025.
        The company reported revenue of USD 42.8 billion in 2025.
        Revenue totaled $42.8 billion in 2025.
        Revenue amounted to $42.8 billion in 2025.
        Revenue increased to $42.8B in 2025.

    The extractor does not use an LLM.
    """

    _NUMBER_PATTERN = (
        r"(?P<number>"
        r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"
        r"|"
        r"\d+(?:\.\d+)?"
        r")"
    )

    _CURRENCY_PATTERN = (
        r"(?P<currency>"
        r"\$|€|£|USD|EUR|GBP"
        r")?"
    )

    _UNIT_PATTERN = (
        r"(?P<unit>"
        r"billion|million|thousand|"
        r"bn|mn|"
        r"B|M|K|"
        r"b|m|k"
        r")?"
    )

    _YEAR_PATTERN = r"(?P<year>20\d{2})"

    # Examples supported:
    #
    # Revenue was $42.8B in 2025
    # Revenue was USD 42.8 billion in 2025
    # Revenue totaled $42.8 billion in 2025
    #
    # The subject is intentionally restricted to a financial phrase rather
    # than allowing the entire sentence to become the subject.
    _NUMERIC_CLAIM_PATTERN = re.compile(
        rf"""
        (?P<subject>
            (?:
                [A-Za-z][A-Za-z0-9&/\-]*
                (?:\s+[A-Za-z][A-Za-z0-9&/\-]*){{0,8}}
            )
        )
        \s+
        (?:
            # Standard forms:
            # "revenue was ..."
            # "revenue totaled ..."
            # "revenue amounted to ..."
            # "revenue increased to ..."
            (?P<verb>
                was|
                were|
                is|
                are|
                reached|
                stood\s+at|
                totaled|
                amounted\s+to|
                increased\s+to|
                decreased\s+to|
                rose\s+to|
                fell\s+to
            )
            \s+
            |
            # Reporting form:
            # "reported revenue of ..."
            # "reported revenue was ..."
            # "reported revenue totaled ..."
            (?P<reported_verb>
                reported
            )
            \s+
        )
        (?:
            # Optional "of" used in:
            # "reported revenue of $42.8B"
            of\s+
        )?
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

    # More precise pattern for the very common:
    #
    # "The company reported revenue of $42.8B in 2025."
    #
    # This pattern is checked first because it prevents the generic pattern
    # from incorrectly treating "The company reported" as the subject.
    _REPORTED_NUMERIC_CLAIM_PATTERN = re.compile(
        rf"""
        (?P<prefix>
            reported
        )
        \s+
        (?P<subject>
            [A-Za-z][A-Za-z0-9&/\-]*
            (?:\s+[A-Za-z][A-Za-z0-9&/\-]*){{0,5}}?
        )
        \s+
        of
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

    # Handles sentences where the subject is followed by "reported":
    #
    # "The company reported revenue of $42.8B in 2025."
    #
    # The previous pattern begins at "reported", while this pattern makes
    # extraction independent of the text before "reported".
    _COMPANY_REPORTED_PATTERN = re.compile(
        rf"""
        \b
        reported
        \s+
        (?P<subject>
            revenue|
            revenues|
            net\s+income|
            gross\s+profit|
            operating\s+income|
            operating\s+loss|
            net\s+loss|
            earnings|
            assets|
            liabilities|
            cash\s+and\s+cash\s+equivalents|
            free\s+cash\s+flow|
            cash\s+flow
        )
        \s+
        of
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
        """
        Extract the first supported financial claim.

        Args:
            text: Text containing a candidate financial claim.
            claim_id: Identifier assigned to the extracted claim.
            source_chunk_id: Retrieval chunk associated with the claim.

        Returns:
            ExtractionResult containing a Claim when extraction succeeds.
        """

        if not text or not text.strip():
            return ExtractionResult(
                claim=None,
                matched_text=None,
            )

        # ---------------------------------------------------------------
        # 1. Most specific reporting pattern
        # ---------------------------------------------------------------
        match = self._COMPANY_REPORTED_PATTERN.search(text)

        if match is not None:
            return self._build_claim(
                match=match,
                claim_id=claim_id,
                source_chunk_id=source_chunk_id,
            )

        # ---------------------------------------------------------------
        # 2. Generic "reported X of VALUE" pattern
        # ---------------------------------------------------------------
        match = self._REPORTED_NUMERIC_CLAIM_PATTERN.search(text)

        if match is not None:
            return self._build_claim(
                match=match,
                claim_id=claim_id,
                source_chunk_id=source_chunk_id,
            )

        # ---------------------------------------------------------------
        # 3. Standard financial claim pattern
        # ---------------------------------------------------------------
        match = self._NUMERIC_CLAIM_PATTERN.search(text)

        if match is None:
            return ExtractionResult(
                claim=None,
                matched_text=None,
            )

        return self._build_claim(
            match=match,
            claim_id=claim_id,
            source_chunk_id=source_chunk_id,
        )

    def _build_claim(
        self,
        match: re.Match,
        claim_id: str,
        source_chunk_id: str | None,
    ) -> ExtractionResult:
        """Build a Claim from a successful regex match."""

        groups = match.groupdict()

        subject = self._clean_subject(
            groups.get("subject"),
        )

        number = groups["number"].replace(
            ",",
            "",
        )

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
    def _clean_subject(
        subject: str | None,
    ) -> str:
        """Normalize the extracted claim subject."""

        if not subject:
            return ""

        cleaned = " ".join(
            subject.split(),
        ).strip()

        return cleaned

    @staticmethod
    def _normalize_unit(
        unit: str | None,
        currency: str | None,
    ) -> str | None:
        """
        Normalize financial magnitude units.

        Currency is intentionally preserved as part of the unit because
        the project's claim schema and tests treat values such as
        '$ billion', '€ billion', and 'USD billion' as distinct units.
        """

        aliases = {
            "b": "billion",
            "bn": "billion",
            "m": "million",
            "mn": "million",
            "k": "thousand",
        }

        normalized_unit = (
            aliases.get(
                unit.lower(),
                unit.lower(),
            )
            if unit
            else None
        )

        if currency and normalized_unit:
            return f"{currency} {normalized_unit}"

        if currency:
            return currency

        return normalized_unit