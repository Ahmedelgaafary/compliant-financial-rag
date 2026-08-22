"""
Entity name matching (exact, later fuzzy).
"""

from typing import Optional

class EntityVerifier:
    """
    Verify that the claimed entity matches the evidence entity.
    Currently uses case‑insensitive exact match.
    Future: fuzzy matching, synonym resolution, CIK/LEI integration.
    """

    def verify(self, claim_entity: str, evidence_entity: str) -> Tuple[bool, str]:
        """
        Returns (is_match, reason).
        Reason: "EXACT_MATCH", "CASE_INSENSITIVE_MATCH", or "MISMATCH".
        """
        if not claim_entity or not evidence_entity:
            return False, "MISSING_ENTITY"

        # Normalise: strip, lower
        claim_norm = claim_entity.strip().lower()
        evidence_norm = evidence_entity.strip().lower()

        if claim_norm == evidence_norm:
            return True, "EXACT_MATCH"

        # Case-insensitive exact match (already done)
        # Could add fuzzy logic here later

        return False, "MISMATCH"
