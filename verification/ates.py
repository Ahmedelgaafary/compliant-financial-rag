"""
Date/period verification with flexible parsing.
"""

import re
from datetime import datetime
from typing import Tuple, Optional, Union


class PeriodType:
    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    DAY = "day"
    UNKNOWN = "unknown"


def parse_period(period_str: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    """
    Parse a period string into a canonical representation.

    Returns:
        (period_type, year, quarter, month)
        Returns (None, None, None, None) if parsing fails.

    Examples:
        "2025" -> ("year", 2025, None, None)
        "Q4 2025" -> ("quarter", 2025, 4, None)
        "2025 Q4" -> ("quarter", 2025, 4, None)
        "FY2025" -> ("year", 2025, None, None)
        "2025-12-31" -> ("day", 2025, None, 12)  # month extracted
        "December 2025" -> ("month", 2025, None, 12)
    """
    if not period_str:
        return None, None, None, None

    s = period_str.strip().lower()

    # Try to extract year (4 digits)
    year_match = re.search(r'\b(20\d{2})\b', s)
    year = int(year_match.group(1)) if year_match else None

    # Quarter detection: Q1-Q4 or Q1 2025, etc.
    quarter_match = re.search(r'[Qq]([1-4])', s)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        period_type = PeriodType.QUARTER
        return period_type, year, quarter, None

    # Month detection: full month names or Jan, Feb, etc.
    months = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    for month_abbr, month_num in months.items():
        if month_abbr in s:
            # Check if it's a full month name or abbreviation
            period_type = PeriodType.MONTH
            return period_type, year, None, month_num

    # FY2025 style
    fy_match = re.search(r'[Ff][Yy](\d{4})', s)
    if fy_match:
        year = int(fy_match.group(1))
        return PeriodType.YEAR, year, None, None

    # Plain year with four digits
    if year is not None and len(s) <= 6:
        return PeriodType.YEAR, year, None, None

    # If we have a year, but no other info, treat as year
    if year is not None:
        return PeriodType.YEAR, year, None, None

    # Unknown
    return None, None, None, None


def compare_periods(
    claim_period: str,
    evidence_period: str
) -> Tuple[bool, str]:
    """
    Compare two period strings and return (is_match, reason).

    Returns:
        (is_match, reason) where reason is one of:
        "EXACT_MATCH", "YEAR_MATCH", "QUARTER_MATCH", "MONTH_MATCH", "MISMATCH"
    """
    if not claim_period or not evidence_period:
        return False, "MISSING"

    cp_type, cp_year, cp_quarter, cp_month = parse_period(claim_period)
    ep_type, ep_year, ep_quarter, ep_month = parse_period(evidence_period)

    if cp_type is None or ep_type is None:
        return False, "UNPARSEABLE"

    # Exact match of type and all available fields
    if cp_type == ep_type and cp_year == ep_year:
        if cp_type == PeriodType.QUARTER and cp_quarter == ep_quarter:
            return True, "EXACT_MATCH"
        if cp_type == PeriodType.MONTH and cp_month == ep_month:
            return True, "EXACT_MATCH"
        if cp_type == PeriodType.YEAR:
            return True, "EXACT_MATCH"

    # Year match is acceptable for many cases
    if cp_year == ep_year:
        return True, "YEAR_MATCH"

    # Otherwise mismatch
    return False, "MISMATCH"


class DateVerifier:
    """
    Verifies that claim period matches evidence period.
    """

    def verify(self, claim_period: str, evidence_period: str) -> Tuple[bool, str]:
        return compare_periods(claim_period, evidence_period)
