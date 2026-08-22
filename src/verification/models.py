from dataclasses import dataclass
from enum import StrEnum


class VerificationStatus(StrEnum):
    """Possible deterministic verification outcomes."""

    VERIFIED = "verified"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class ClaimType(StrEnum):
    """Supported financial claim categories."""

    NUMERIC = "numeric"
    DATE = "date"
    ENTITY = "entity"
    TEXT = "text"


@dataclass(frozen=True)
class Claim:
    """A claim that requires verification."""

    claim_id: str
    claim_type: ClaimType
    subject: str
    value: str
    unit: str | None = None
    period: str | None = None
    source_chunk_id: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    """Result produced by the deterministic verifier."""

    claim_id: str
    status: VerificationStatus
    reason: str
    confidence: float
    evidence_chunk_id: str | None = None