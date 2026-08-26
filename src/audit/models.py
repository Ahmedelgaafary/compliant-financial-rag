# src/audit/models.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class AuditStatus(str, Enum):
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    REJECTED = "REJECTED"


class ReviewDecision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_MORE_INFO = "NEEDS_MORE_INFO"
    ESCALATE = "ESCALATE"


@dataclass
class AuditRecord:
    """Complete audit trail for a financial query."""
    audit_id: str
    timestamp: datetime
    user_query: str
    claim: str
    verification_status: str
    verification_reason: str
    risk_level: str
    evidence: List[dict]
    document_id: str
    document_sha256: str
    page_number: int
    # Added with defaults to avoid breaking existing code
    risk_assessment: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Review fields (populated later)
    reviewer: Optional[str] = None
    review_decision: Optional[ReviewDecision] = None
    review_notes: Optional[str] = None
    review_timestamp: Optional[datetime] = None
    status: AuditStatus = AuditStatus.PENDING

    # Additional metadata
    confidence_score: Optional[float] = None
    risk_score: Optional[float] = None
    triggers: List[str] = field(default_factory=list)
    verification_results: List[dict] = field(default_factory=list)