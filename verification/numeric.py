"""
Numeric claim verification with normalisation and tolerance.
"""

import re
from typing import Tuple, Optional, Union
from decimal import Decimal, getcontext
import math

# Set high precision for Decimal if needed
getcontext().prec = 28

# Unit multipliers (scale factor)
UNIT_MAP = {
    "": 1.0,
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "t": 1e12,
}


def normalise_numeric(value_str: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Parse a financial numeric string into (numeric_value, unit, value_type).

    Parameters:
        value_str: e.g., "$42.8 billion", "42.8B", "42,800 million", "42.8%"

    Returns:
        (float_value, unit, type) where type is "amount" or "percent".
        If parsing fails, returns (None, None, None).

    Examples:
        "$42.8 billion" -> (42.8, "billion", "amount")
        "42.8B" -> (42.8, "billion", "amount")
        "42,800 million" -> (42800.0, "million", "amount")  # value before scaling
        "42.8%" -> (42.8, "%", "percent")
    """
    if not value_str:
        return None, None, None

    # Remove commas and whitespace
    s = re.sub(r'[,\s]', '', value_str.strip())

    # Detect percent sign
    if s.endswith('%'):
        s = s[:-1]
        value_type = "percent"
        unit = "%"
    else:
        value_type = "amount"
        unit = None

    # Remove currency symbols ($, €, £, etc.)
    s = re.sub(r'^[^\d\-\.]+', '', s)  # remove leading non-digit (except minus)

    # Extract numeric part and unit part
    # Pattern: (number)(optional unit)
    # Number may include decimal point
    match = re.match(r'^([-+]?\d*\.?\d+)([a-zA-Z%]*)$', s)
    if not match:
        return None, None, None

    num_str, unit_str = match.groups()
    try:
        num_val = float(num_str)
    except ValueError:
        return None, None, None

    # Determine unit
    unit_lower = unit_str.lower()
    if unit_lower in UNIT_MAP:
        unit = unit_lower
    elif unit_lower == "":
        unit = ""
    else:
        # unknown unit, treat as plain number
        unit = ""

    return num_val, unit, value_type


def compare_numeric(
    claim_val: float,
    claim_unit: str,
    claim_type: str,  # "amount" or "percent"
    evidence_val: float,
    evidence_unit: str,
    evidence_type: str,
    tolerance: float = 0.01  # relative tolerance (1%)
) -> Tuple[bool, float, float]:
    """
    Compare a claim value with an evidence value after scaling to a common unit.

    For amounts: scale both to raw units using UNIT_MAP.
    For percentages: compare as plain numbers (no scaling).

    Returns:
        (is_match, scaled_claim, scaled_evidence)
    """
    # For percentages, just compare numbers (no unit scaling)
    if claim_type == "percent" or evidence_type == "percent":
        # Both should be percent; if one is not, treat as mismatch
        if claim_type != evidence_type:
            return False, claim_val, evidence_val
        claim_scaled = claim_val
        evidence_scaled = evidence_val
    else:
        # Scale both to raw value (assuming base unit is "1")
        claim_scale = UNIT_MAP.get(claim_unit, 1.0) if claim_unit else 1.0
        evidence_scale = UNIT_MAP.get(evidence_unit, 1.0) if evidence_unit else 1.0
        claim_scaled = claim_val * claim_scale
        evidence_scaled = evidence_val * evidence_scale

    # If both are zero, consider match
    if claim_scaled == 0 and evidence_scaled == 0:
        return True, claim_scaled, evidence_scaled

    # Relative difference
    if abs(claim_scaled) > 1e-9 and abs(evidence_scaled) > 1e-9:
        diff = abs(claim_scaled - evidence_scaled) / max(abs(claim_scaled), abs(evidence_scaled))
    else:
        diff = abs(claim_scaled - evidence_scaled)

    is_match = diff <= tolerance
    return is_match, claim_scaled, evidence_scaled


class NumericVerifier:
    """
    Verifies numeric claims against evidence.
    """

    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance

    def verify(
        self,
        claim_value: str,
        claim_unit: Optional[str],
        claim_type: str,  # "amount" or "percent"
        evidence_texts: list,
    ) -> Tuple[bool, Optional[str], float, float]:
        """
        Verify a numeric claim against a list of evidence text snippets.

        Assumes that evidence_texts are strings that may contain numbers.
        For each evidence, we try to extract a numeric value and unit.
        If multiple evidence match, we take the first that matches.

        Returns:
            (is_match, matched_unit, claim_scaled, evidence_scaled)
            If no evidence can be parsed, returns (False, None, 0, 0)
        """
        # Normalise the claim
        claim_val, claim_unit_norm, claim_val_type = normalise_numeric(claim_value)
        if claim_val is None:
            raise ValueError(f"Could not parse claim value: {claim_value}")

        # If claim_type not provided, use detected type
        if claim_type is None:
            claim_type = claim_val_type

        for evidence in evidence_texts:
            # Extract numeric from evidence text
            ev_val, ev_unit, ev_type = normalise_numeric(evidence)
            if ev_val is None:
                continue

            is_match, c_scaled, e_scaled = compare_numeric(
                claim_val,
                claim_unit_norm,
                claim_type,
                ev_val,
                ev_unit,
                ev_type,
                self.tolerance
            )
            if is_match:
                return True, ev_unit, c_scaled, e_scaled

        # No matching evidence found
        return False, None, 0.0, 0.0
