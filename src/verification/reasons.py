from enum import StrEnum


class VerificationReason(StrEnum):
    """Machine-readable verification reasons."""

    NUMERIC_MATCH = "numeric_match"
    NUMERIC_MISMATCH = "numeric_mismatch"

    PERIOD_MATCH = "period_match"
    PERIOD_MISMATCH = "period_mismatch"

    ENTITY_MATCH = "entity_match"
    ENTITY_MISMATCH = "entity_mismatch"

    EVIDENCE_MISSING = "evidence_missing"
    EVIDENCE_CONTRADICTS = "evidence_contradicts"

    UNSUPPORTED_CLAIM = "unsupported_claim"