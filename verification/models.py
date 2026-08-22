"""
Data models for claims and verification results.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Any


class ClaimType(Enum):
    NUMERIC = auto()
    DATE = auto()
    ENTITY = auto()
    TEXT = auto()
    # Future: PERCENTAGE, CURRENCY, RATIO, COUNT, FINANCIAL_METRIC, COMPARISON


class VerificationStatus(Enum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


# Reason codes defined in reasons.py, but we import them here for convenience.
# We'll keep enum separate to avoid circular imports.
# We'll define VerificationReason in reasons.py and import here if needed.
# For clarity, we'll define it here and extend in reasons.py (optional).

class VerificationReason(Enum):
    NUMERIC_MATCH = "NUMERIC_MATCH"
    NUMERIC_MISMATCH = "NUMERIC_MISMATCH"
    PERIOD_MATCH = "PERIOD_MATCH"
    PERIOD_MISMATCH = "PERIOD_MISMATCH"
    ENTITY_MATCH = "ENTITY_MATCH"
    ENTITY_MISMATCH = "ENTITY_MISMATCH"
    EVIDENCE_MISSING = "EVIDENCE_MISSING"
    EVIDENCE_CONTRADICTS = "EVIDENCE_CONTRADICTS"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"


@dataclass
class Claim:
    """
    A financial claim extracted from an LLM response or user question.
    """
    claim_id: str
    claim_type: ClaimType
    subject: str          
    # e.g., "revenue", "net income", "total assets"
    value: Optional[str] = None         
    # raw string value, e.g., "42.8"
    unit: Optional[str] = None           
    # e.g., "billion", "million", "%"
    period: Optional[str] = None        
     # e.g., "2025", "Q4 2025", "FY2024"
    entity: Optional[str] = None        
    # company name
    source_chunk_id: Optional[str] = None 
  # if claim came from a specific chunk
    metadata: dict = field(default_factory=dict) 
  # extra info


@dataclass
class VerificationResult:
    """
    Result of verifying a claim against evidence.
    """
    claim_id: str
    status: VerificationStatus
    reason: VerificationReason
    confidence: float  
    # 0.0 to 1.0, based on evidence support
    evidence_chunk_id: Optional[str] = None  
   # the chunk that supports the decision
    details: dict = field(default_factory=dict) 
 # extra info (e.g., matched value, period)
