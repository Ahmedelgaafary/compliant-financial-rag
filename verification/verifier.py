"""
Orchestrates all verification components.
"""

from typing import List, Optional, Dict, Any
from .models import Claim, ClaimType, VerificationStatus, VerificationReason, VerificationResult
from .numeric import NumericVerifier, normalise_numeric
from .dates import DateVerifier, parse_period
from .entities import EntityVerifier
from .provenance import ProvenanceVerifier
from .contradiction import ContradictionDetector

# Assume evidence is a dict with fields: chunk_id, document_id, text, page_number, section, document_sha256, score, retrieval_method

class Verifier:
    """
    Main verification engine.
    """

    def __init__(self, numeric_tolerance: float = 0.01):
        self.numeric_verifier = NumericVerifier(tolerance=numeric_tolerance)
        self.date_verifier = DateVerifier()
        self.entity_verifier = EntityVerifier()
        self.provenance_verifier = ProvenanceVerifier()
        self.contradiction_detector = ContradictionDetector()

    def verify_claim(
        self,
        claim: Claim,
        evidence_list: List[Dict[str, Any]]
    ) -> VerificationResult:
        """
        Verify a single claim against a list of evidence items.
        """
        # 1. Check provenance
        for evidence in evidence_list:
            is_valid, missing = self.provenance_verifier.verify(evidence)
            if not is_valid:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status=VerificationStatus.REJECTED,
                    reason=VerificationReason.EVIDENCE_MISSING,
                    confidence=0.0,
                    details={"missing_fields": missing}
                )

        # 2. Check entity
        if claim.entity:
            # Use the first evidence's entity (or document metadata)
            # In a real system, we would extract entity from document metadata
            evidence_entity = None
            for ev in evidence_list:
                # Assume evidence has a 'entity' field or we can get from document metadata
                # For now, we just take the first evidence's document_id as a placeholder
                evidence_entity = ev.get('entity', None) or ev.get('document_id', '')
                break
            if evidence_entity:
                is_match, reason = self.entity_verifier.verify(claim.entity, evidence_entity)
                if not is_match:
                    return VerificationResult(
                        claim_id=claim.claim_id,
                        status=VerificationStatus.REJECTED,
                        reason=VerificationReason.ENTITY_MISMATCH,
                        confidence=0.0,
                        details={"reason": reason}
                    )

        # 3. Contradiction detection
        has_contradiction, contradiction_desc = self.contradiction_detector.detect_contradiction(claim, evidence_list)
        if has_contradiction:
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.EVIDENCE_CONTRADICTS,
                confidence=0.5,
                details={"contradiction": contradiction_desc}
            )

        # 4. Claim type specific verification
        if claim.claim_type == ClaimType.NUMERIC:
            # Extract text from evidence
            evidence_texts = [ev.get('text', '') for ev in evidence_list if ev.get('text')]
            if not evidence_texts:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status=VerificationStatus.INCONCLUSIVE,
                    reason=VerificationReason.EVIDENCE_MISSING,
                    confidence=0.0,
                    details={"reason": "No evidence text found"}
                )

            # Normalise claim value
            claim_val, claim_unit, claim_type = normalise_numeric(claim.value)
            if claim_val is None:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status=VerificationStatus.REJECTED,
                    reason=VerificationReason.UNSUPPORTED_CLAIM,
                    confidence=0.0,
                    details={"reason": "Could not parse claim numeric value"}
                )

            # Use numeric verifier
            is_match, matched_unit, claim_scaled, ev_scaled = self.numeric_verifier.verify(
                claim_value=claim.value,
                claim_unit=claim.unit,
                claim_type=claim_type,
                evidence_texts=evidence_texts
            )
            if is_match:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status=VerificationStatus.VERIFIED,
                    reason=VerificationReason.NUMERIC_MATCH,
                    confidence=1.0,
                    evidence_chunk_id=evidence_list[0].get('chunk_id'),
                    details={"matched_value": ev_scaled, "claim_scaled": claim_scaled, "unit": matched_unit}
                )
            else:
                # Try to see if any evidence exists but doesn't match
                # If we have evidence but no match, it's a mismatch
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status=VerificationStatus.REJECTED,
                    reason=VerificationReason.NUMERIC_MISMATCH,
                    confidence=0.0,
                    details={"reason": "No evidence matches the claim value"}
                )

        elif claim.claim_type == ClaimType.DATE:
            # Similar logic for dates
            evidence_texts = [ev.get('text', '') for ev in evidence_list if ev.get('text')]
            if not evidence_texts:
                return VerificationResult(
                    claim_id=claim.claim_id,
                    status=VerificationStatus.INCONCLUSIVE,
                    reason=VerificationReason.EVIDENCE_MISSING,
                    confidence=0.0
                )
            # Use first evidence that contains a date
            for text in evidence_texts:
                is_match, reason = self.date_verifier.verify(claim.period or "", text)
                if is_match:
                    return VerificationResult(
                        claim_id=claim.claim_id,
                        status=VerificationStatus.VERIFIED,
                        reason=VerificationReason.PERIOD_MATCH,
                        confidence=1.0,
                        evidence_chunk_id=evidence_list[0].get('chunk_id'),
                        details={"matched_period": text}
                    )
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.REJECTED,
                reason=VerificationReason.PERIOD_MISMATCH,
                confidence=0.0
            )

        # Add other claim types (ENTITY, TEXT) as needed
        else:
            # For unsupported claim types, treat as inconclusive
            return VerificationResult(
                claim_id=claim.claim_id,
                status=VerificationStatus.INCONCLUSIVE,
                reason=VerificationReason.UNSUPPORTED_CLAIM,
                confidence=0.0,
                details={"reason": "Claim type not yet supported"}
            )
