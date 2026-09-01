from dataclasses import dataclass

from src.verification.models import Claim, VerificationResult
from src.verification.provenance import EvidenceProvenance


@dataclass(frozen=True)
class VerificationPipelineResult:
    """Result produced by deterministic claim verification."""

    claim: Claim
    verification: VerificationResult
    provenance: EvidenceProvenance


class VerificationPipeline:
    """Coordinate deterministic claim verification."""

    def __init__(self, claim_verifier) -> None:
        self._claim_verifier = claim_verifier

    def process(
        self,
        claim: Claim,
        evidence_text: str,
        provenance: EvidenceProvenance,
    ) -> VerificationPipelineResult:
        """Verify a claim against supplied evidence and provenance.

        Provenance is passed into the verifier itself so that a claim
        cannot become VERIFIED independently of the evidence citation
        that supports it.
        """

        verification = self._claim_verifier.verify(
            claim=claim,
            evidence_text=evidence_text,
            provenance=provenance,
        )

        return VerificationPipelineResult(
            claim=claim,
            verification=verification,
            provenance=provenance,
        )