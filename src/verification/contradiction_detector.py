import re

from src.verification.models import (
    Claim,
    VerificationResult,
    VerificationStatus,
)
from src.verification.reasons import VerificationReason


class ContradictionDetector:
    """Detect explicit contradictions between a claim and evidence."""

    _NEGATION_PATTERNS = (
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bdenied\b",
        r"\bdenies\b",
        r"\bdenial\b",
        r"\bincorrect\b",
        r"\binaccurate\b",
        r"\bfalse\b",
        r"\bdoes not\b",
        r"\bdid not\b",
        r"\bwas not\b",
        r"\bwere not\b",
        r"\bis not\b",
        r"\bare not\b",
    )

    def verify(
        self,
        claim: Claim,
        evidence: str | None,
    ) -> VerificationResult:
        """Detect an explicit contradiction in evidence."""

        if not evidence or not evidence.strip():
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.EVIDENCE_MISSING,
                confidence=0.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        normalized_claim = self._normalize(claim.value)
        normalized_evidence = self._normalize(evidence)

        if not normalized_claim:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=0.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        if self._contains_explicit_contradiction(
            normalized_claim,
            normalized_evidence,
        ):
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.REJECTED,
                reason=VerificationReason.EVIDENCE_CONTRADICTS,
                confidence=1.0,
                evidence_chunk_id=claim.source_chunk_id,
            )

        return VerificationResult(
            claim_id=claim.claim_id,
            status=VerificationStatus.INCONCLUSIVE,
            reason=VerificationReason.UNSUPPORTED_CLAIM,
            confidence=0.0,
            evidence_chunk_id=claim.source_chunk_id,
        )

    def _contains_explicit_contradiction(
        self,
        claim_value: str,
        evidence: str,
    ) -> bool:
        """Return whether evidence explicitly negates the claim."""

        if claim_value not in evidence:
            return False

        claim_index = evidence.find(claim_value)

        before = evidence[:claim_index]
        after = evidence[claim_index + len(claim_value):]

        context = f"{before[-80:]} {after[:80]}"

        return any(
            re.search(pattern, context)
            for pattern in self._NEGATION_PATTERNS
        )

    @staticmethod
    def _normalize(value: str) -> str:
        """Normalize text for deterministic comparison."""

        return " ".join(value.casefold().split())