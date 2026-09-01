# src/api/schemas.py

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""

    user_query: str = Field(..., description="Financial question to ask the agent.")


class EvidenceResponse(BaseModel):
    """A single piece of retrieved evidence."""

    chunk_id: str
    document_id: str
    text: str
    score: Optional[float] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    document_sha256: Optional[str] = None
    retrieval_method: Optional[str] = None
    document_name: Optional[str] = None
    document_path: Optional[str] = None


class VerificationResponse(BaseModel):
    """A single verification result."""

    claim_id: str
    status: str
    reason: str
    confidence: Optional[float] = None
    evidence_chunk_id: Optional[str] = None


class RiskResponse(BaseModel):
    """Risk assessment response."""

    risk_score: float
    risk_level: str
    triggers: List[str]
    recommended_action: str


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""

    final_answer: str
    status: str
    should_route_to_audit: bool
    audit_id: Optional[str] = None
    evidence: List[EvidenceResponse] = []
    verification_results: List[VerificationResponse] = []
    risk_assessment: Optional[RiskResponse] = None
    claims_count: Optional[int] = None
    query_analysis: Optional[Dict[str, Any]] = None


class AuditDecisionRequest(BaseModel):
    """Request model for submitting an audit decision."""

    decision: str = Field(..., description="APPROVE or REJECT")
    notes: Optional[str] = Field(None, description="Review notes")
    reviewer: str = Field(..., description="Reviewer name or ID")


class AuditDecisionResponse(BaseModel):
    """Response model for an audit decision."""

    audit_id: str
    status: str
    review_decision: Optional[str]
    reviewer: Optional[str]
    review_notes: Optional[str]
    review_timestamp: Optional[datetime]


class AuditResponse(BaseModel):
    """Full audit record response."""

    audit_id: str
    timestamp: datetime
    user_query: str
    claim: str
    verification_status: str
    verification_reason: str
    risk_level: str
    evidence: List[Dict[str, Any]]
    provenance: Dict[str, Any]
    claim_id: Optional[str]
    document_id: Optional[str]
    document_sha256: Optional[str]
    page_number: Optional[int]
    risk_assessment: Optional[Dict[str, Any]]
    created_at: datetime
    reviewer: Optional[str]
    review_decision: Optional[str]
    review_notes: Optional[str]
    review_timestamp: Optional[datetime]
    status: str
    confidence_score: Optional[float]
    risk_score: Optional[float]
    triggers: Optional[List[str]]
    verification_results: Optional[List[Dict[str, Any]]]


class DocumentResponse(BaseModel):
    """Response model for document endpoints."""

    name: str
    path: str
    size: int
    modified: float
    directory: str
    company: Optional[str] = None


class CompanyInfoResponse(BaseModel):
    """Response model for company info endpoint."""

    name: str
    variations: List[str]
    documents: List[DocumentResponse]