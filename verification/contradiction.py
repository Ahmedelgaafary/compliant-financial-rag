"""
Detect contradictions across multiple evidence pieces.
"""

from typing import List, Dict, Any, Optional
from .numeric import normalise_numeric, compare_numeric
from .dates import parse_period
from .models import Claim, ClaimType

class ContradictionDetector:
    """
    Check if multiple evidence items disagree on the same claim's value.
    """

    def detect_contradiction(
        self,
        claim: Claim,
        evidence_values: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        For a given claim, examine all evidence values (extracted from evidence texts).
        Returns (has_contradiction, description).

        If all evidence that can be parsed agree, return False (no contradiction).
        If any disagreement, return True.
        """
        if not evidence_values or len(evidence_values) < 2:
            return False, None

        # Normalise each evidence value based on claim type
        normalised_values = []
        for ev in evidence_values:
            # Extract the relevant numeric/date/entity from the evidence
            # For simplicity, we assume evidence_values is a list of strings or dicts with 'value' field
            # In real usage, we would extract from evidence chunks.
            # Here we assume each ev has a 'text' field from which we can extract.
            text = ev.get('text', '')
            if not text:
                continue

            # Try to parse as numeric if claim is numeric
            if claim.claim_type == ClaimType.NUMERIC:
                val, unit, vtype = normalise_numeric(text)
                if val is not None:
                    normalised_values.append((val, unit, vtype))
            elif claim.claim_type == ClaimType.DATE:
                # Parse period
                ptype, year, quarter, month = parse_period(text)
                if ptype is not None:
                    normalised_values.append((ptype, year, quarter, month))
            # Add other types later

        if len(normalised_values) < 2:
            return False, None

        # Check if all values are equal (within tolerance for numeric)
        # For numeric, compare pairwise
        if claim.claim_type == ClaimType.NUMERIC:
            # Compare all pairs
            for i in range(len(normalised_values)):
                for j in range(i+1, len(normalised_values)):
                    v1, u1, t1 = normalised_values[i]
                    v2, u2, t2 = normalised_values[j]
                    is_match, _, _ = compare_numeric(v1, u1, t1, v2, u2, t2, tolerance=0.01)
                    if not is_match:
                        return True, f"Numeric contradiction between evidence {i+1} and {j+1}"
        elif claim.claim_type == ClaimType.DATE:
            # Compare periods
            for i in range(len(normalised_values)):
                for j in range(i+1, len(normalised_values)):
                    p1, y1, q1, m1 = normalised_values[i]
                    p2, y2, q2, m2 = normalised_values[j]
                    # Simple equality check on year, quarter, month
                    if (y1 != y2) or (q1 != q2) or (m1 != m2):
                        return True, f"Date contradiction between evidence {i+1} and {j+1}"

        return False, None
