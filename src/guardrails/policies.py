import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class GuardrailPolicies:
    """Central configuration for guardrail behaviour."""

    # ==========================================
    # READ FROM ENVIRONMENT WITH FALLBACKS
    # ==========================================
    
    # Confidence thresholds - CRITICAL FIX
    min_retrieval_confidence: float = float(
        os.environ.get("MIN_RETRIEVAL_CONFIDENCE", "0.01")
    )
    min_overall_confidence: float = float(
        os.environ.get("MIN_OVERALL_CONFIDENCE", "0.70")
    )
    min_evidence_chunks: int = int(
        os.environ.get("MIN_EVIDENCE_CHUNKS", "1")
    )

    # Risk thresholds
    risk_low_threshold: float = float(
        os.environ.get("LOW_RISK_THRESHOLD", "0.20")
    )
    risk_medium_threshold: float = float(
        os.environ.get("MEDIUM_RISK_THRESHOLD", "0.50")
    )
    risk_high_threshold: float = float(
        os.environ.get("HIGH_RISK_THRESHOLD", "0.80")
    )
    max_risk_score: float = float(
        os.environ.get("MAX_RISK_SCORE", "0.7")
    )

    # Risk increments
    risk_increment_rejected: float = float(
        os.environ.get("RISK_INCREMENT_REJECTED", "0.5")
    )
    risk_increment_inconclusive: float = float(
        os.environ.get("RISK_INCREMENT_INCONCLUSIVE", "0.1")
    )
    risk_increment_contradiction: float = float(
        os.environ.get("RISK_INCREMENT_CONTRADICTION", "0.3")
    )
    risk_increment_low_confidence: float = float(
        os.environ.get("RISK_INCREMENT_LOW_CONFIDENCE", "0.6")
    )
    risk_increment_missing_provenance: float = float(
        os.environ.get("RISK_INCREMENT_MISSING_PROVENANCE", "0.8")
    )
    risk_increment_no_evidence: float = float(
        os.environ.get("RISK_INCREMENT_NO_EVIDENCE", "0.8")
    )
    risk_increment_insufficient_evidence: float = float(
        os.environ.get("RISK_INCREMENT_INSUFFICIENT_EVIDENCE", "0.2")
    )
    risk_increment_numeric_mismatch: float = float(
        os.environ.get("RISK_INCREMENT_NUMERIC_MISMATCH", "0.4")
    )

    # Guardrail behavior
    block_on_numeric_mismatch: bool = (
        os.environ.get("BLOCK_ON_NUMERIC_MISMATCH", "False").lower() == "true"
    )
    allow_unsupported_claims: bool = (
        os.environ.get("ALLOW_UNSUPPORTED_CLAIMS", "False").lower() == "true"
    )
    include_disclaimer_on_low_confidence: bool = (
        os.environ.get("INCLUDE_DISCLAIMER_ON_LOW_CONFIDENCE", "True").lower() == "true"
    )
    max_contradictions_before_audit: int = int(
        os.environ.get("MAX_CONTRADICTIONS_BEFORE_AUDIT", "1")
    )

    # Allowed query types
    allowed_query_types: List[str] = field(
        default_factory=lambda: ["numeric", "comparison", "trend", "entity", "period"]
    )

    # Forbidden entities
    forbidden_entities: List[str] = field(
        default_factory=lambda: [
            e.strip() for e in os.environ.get("FORBIDDEN_ENTITIES", "").split(",") 
            if e.strip()
        ]
    )

    # RRF configuration
    rrf_k: int = int(os.environ.get("RRF_K", "60"))

    # Max evidence chunks
    max_evidence_chunks: int = int(
        os.environ.get("MAX_EVIDENCE_CHUNKS", "5")
    )

    def is_query_allowed(self, query_type: str) -> bool:
        return not self.allowed_query_types or query_type in self.allowed_query_types

    def get_risk_level(self, risk_score: float) -> str:
        if risk_score >= self.risk_high_threshold:
            return "HIGH"
        if risk_score >= self.risk_medium_threshold:
            return "MEDIUM"
        return "LOW"