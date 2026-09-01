"""Backward-compatible façade for deterministic claim verification."""

from src.verification.models import (
    Claim,
    VerificationResult,
)
from src.verification.verification_engine import VerificationEngine


class ClaimVerifier:
    """Delegate deterministic verification to VerificationEngine."""

    def __init__(
        self,
        verification_engine: VerificationEngine | None = None,
    ) -> None:
        self._engine = (
            verification_engine
            or VerificationEngine()
        )

    def verify(
        self,
        claim: Claim,
        evidence_text: str,
    ) -> VerificationResult:
        """Verify a claim through the deterministic engine."""

        return self._engine.verify(
            claim=claim,
            evidence_text=evidence_text,
        )