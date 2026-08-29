# src/api/schemas.py

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.audit.models import AuditStatus, ReviewDecision


class QueryRequest(BaseModel):
    """Request payload for submitting a financial query."""

    model_config = ConfigDict(extra="forbid")

    user_query: str = Field(
        ...,
        min_length=1,
        description="Financial question submitted by the user.",
    )


class EvidenceResponse(BaseModel):
    """Evidence returned with a verified or reviewed response."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    document_id: str
    text: str
    score: float | None = None
    page_number: int | None = None
    section: str | None = None
    document_sha256: str | None = None
    retrieval_method: str | None = None


class VerificationResponse(BaseModel):
    """Deterministic verification result exposed by the API."""

    model_config = ConfigDict(extra="allow")

    claim_id: str
    status: str
    reason: str
    confidence: float | None = None
    evidence_chunk_id: str | None = None


class RiskResponse(BaseModel):
    """Risk assessment exposed by the API."""

    model_config = ConfigDict(extra="allow")

    risk_score: float
    risk_level: str
    triggers: list[str] = Field(default_factory=list)
    recommended_action: str


class QueryResponse(BaseModel):
    """Final response returned to a user query."""

    model_config = ConfigDict(extra="allow")

    final_answer: str
    status: str
    should_route_to_audit: bool = False
    audit_id: str | None = None

    evidence: list[EvidenceResponse] = Field(
        default_factory=list,
    )

    verification_results: list[VerificationResponse] = Field(
        default_factory=list,
    )

    risk_assessment: RiskResponse | None = None


class AuditResponse(BaseModel):
    """Audit case exposed through the API."""

    model_config = ConfigDict(from_attributes=True)

    audit_id: str
    timestamp: datetime
    user_query: str

    claim: str
    verification_status: str
    verification_reason: str

    risk_level: str

    evidence: list[dict] = Field(
        default_factory=list,
    )

    provenance: list[dict] = Field(
        default_factory=list,
    )

    claim_id: str = ""
    document_id: str = ""
    document_sha256: str = ""
    page_number: int = 0

    risk_assessment: str = ""

    created_at: datetime

    reviewer: str | None = None
    review_decision: ReviewDecision | None = None
    review_notes: str | None = None
    review_timestamp: datetime | None = None

    status: AuditStatus = AuditStatus.PENDING

    confidence_score: float | None = None
    risk_score: float | None = None

    triggers: list[str] = Field(
        default_factory=list,
    )

    verification_results: list[dict] = Field(
        default_factory=list,
    )


class AuditDecisionRequest(BaseModel):
    """Request payload for submitting a human audit decision."""

    model_config = ConfigDict(extra="forbid")

    reviewer: str = Field(
        ...,
        min_length=1,
        description="Identifier of the human reviewer.",
    )

    decision: ReviewDecision

    notes: str | None = None


class AuditDecisionResponse(BaseModel):
    """Result returned after applying a human audit decision."""

    model_config = ConfigDict(extra="allow")

    audit_id: str
    status: AuditStatus
    review_decision: ReviewDecision
    reviewer: str
    review_notes: str | None = None
    review_timestamp: datetime | None = None