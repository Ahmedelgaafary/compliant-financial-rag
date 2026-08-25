# src/verification/verifier.py
from typing import List

from src.retrieval.models import RetrievalResult
from src.verification.models import Claim, VerificationResult


class Verifier:
    """Deterministic verifier – checks if claim text appears in evidence."""

    def verify_claims(
        self,
        claims: List[Claim],
        evidence: List[RetrievalResult],
    ) -> List[VerificationResult]:
        # Simple stub – replace with real logic.
        results = []
        for claim in claims:
            found = any(claim.text.lower() in ev.text.lower() for ev in evidence)
            status = "VERIFIED" if found else "INCONCLUSIVE"
            reason = "Evidence found" if found else "No direct evidence"
            results.append(
                VerificationResult(
                    claim_id=claim.claim_id,
                    status=status,
                    reason=reason,
                    confidence=0.9 if found else 0.1,
                    evidence_chunk_id=evidence[0].chunk_id if evidence else None,
                )
            )
        return results