from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason


class EntityVerifier:
    """Deterministically verify entity claims against evidence."""

    def verify(
        self,
        claim: Claim,
        evidence: str | None,
        evidence_chunk_id: str | None = None,
    ) -> VerificationResult:
        """Verify whether the claimed entity appears in evidence."""

        resolved_chunk_id = (
            evidence_chunk_id
            or claim.source_chunk_id
        )

        if claim.claim_type != ClaimType.ENTITY:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=0.0,
                evidence_chunk_id=resolved_chunk_id,
                claim_type=claim.claim_type,
            )

        if not evidence or not evidence.strip():
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.EVIDENCE_MISSING,
                confidence=0.0,
                evidence_chunk_id=resolved_chunk_id,
                claim_type=claim.claim_type,
            )

        normalized_claim = self._normalize(
            claim.value
        )

        if not normalized_claim:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.REJECTED,
                reason=VerificationReason.ENTITY_MISMATCH,
                confidence=1.0,
                evidence_chunk_id=resolved_chunk_id,
                claim_type=claim.claim_type,
            )

        normalized_evidence = self._normalize(
            evidence
        )

        if normalized_claim in normalized_evidence:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.VERIFIED,
                reason=VerificationReason.ENTITY_MATCH,
                confidence=1.0,
                evidence_chunk_id=resolved_chunk_id,
                claim_type=claim.claim_type,
            )

        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.REJECTED,
            reason=VerificationReason.ENTITY_MISMATCH,
            confidence=1.0,
            evidence_chunk_id=resolved_chunk_id,
            claim_type=claim.claim_type,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        """Normalize text for deterministic entity comparison."""

        return " ".join(
            value.casefold().split()
        )