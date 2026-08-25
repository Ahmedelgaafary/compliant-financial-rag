"""
Purpose: Define configurable policies – thresholds, allowed query types, risk levels, and behaviour rules.
"""
# src/guardrails/policies.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class GuardrailPolicies:
    """
    Central configuration for guardrail behaviour.
    """

    # Confidence thresholds
    min_overall_confidence: float = 0.7
    min_retrieval_confidence: float = 0.5

    # Risk thresholds
    risk_low_threshold: float = 0.2
    risk_medium_threshold: float = 0.5
    risk_high_threshold: float = 0.8

    # Allowed query types – if empty, all are allowed
    allowed_query_types: List[str] = field(default_factory=lambda: [
        "numeric", "comparison", "trend", "entity", "period"
    ])

    # Whether to allow generation of claims not directly supported by evidence
    allow_unsupported_claims: bool = False

    # Maximum number of evidence chunks to consider
    max_evidence_chunks: int = 5

    # If contradictions exceed this, force human review
    max_contradictions_before_audit: int = 1

    # If a critical verification failure (e.g., numeric mismatch) occurs,
    # whether to block output entirely
    block_on_numeric_mismatch: bool = True

    # Whether to include disclaimers when confidence is below threshold
    include_disclaimer_on_low_confidence: bool = True

    # List of forbidden entities (e.g., sensitive names)
    forbidden_entities: List[str] = field(default_factory=list)

    def is_query_allowed(self, query_type: str) -> bool:
        return not self.allowed_query_types or query_type in self.allowed_query_types

    def get_risk_level(self, risk_score: float) -> str:
        if risk_score >= self.risk_high_threshold:
            return "HIGH"
        if risk_score >= self.risk_medium_threshold:
            return "MEDIUM"
        return "LOW"