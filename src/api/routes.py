# src/api/routes.py

from typing import List

from fastapi import APIRouter, HTTPException, status

from src.agent.node import _detect_company_from_text, _find_pdf_documents
from src.agent.workflow import run_agent
from src.api.schemas import (
    AuditDecisionRequest,
    AuditDecisionResponse,
    AuditResponse,
    DocumentResponse,
    EvidenceResponse,
    QueryRequest,
    QueryResponse,
    RiskResponse,
    VerificationResponse,
)
from src.audit.models import AuditRecord
from src.audit.review_service import ReviewService

router = APIRouter()

# Keep one service instance so the in-memory audit queue is shared
# across API requests.
_review_service = ReviewService()


def _get_state_value(state, name: str, default=None):
    """Read a value from either AgentState or a LangGraph state mapping."""

    if isinstance(state, dict):
        return state.get(name, default)

    return getattr(state, name, default)


def _serialize_evidence(state) -> list[EvidenceResponse]:
    """Convert retrieval results into API response models."""

    results = _get_state_value(
        state,
        "retrieval_results",
        [],
    )

    evidence: list[EvidenceResponse] = []

    for result in results:
        evidence.append(
            EvidenceResponse(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                score=getattr(result, "score", None),
                page_number=getattr(result, "page_number", None),
                section=getattr(result, "section", None),
                document_sha256=getattr(
                    result,
                    "document_sha256",
                    None,
                ),
                retrieval_method=getattr(
                    result,
                    "retrieval_method",
                    None,
                ),
                document_name=getattr(result, "document_name", None),
                document_path=getattr(result, "document_path", None),
            )
        )

    return evidence


def _serialize_verification_results(state) -> list[VerificationResponse]:
    """Convert verification results into API response models."""

    results = _get_state_value(
        state,
        "verification_results",
        [],
    )

    return [
        VerificationResponse(
            claim_id=result.claim_id,
            status=str(result.status),
            reason=result.reason,
            confidence=getattr(
                result,
                "confidence",
                None,
            ),
            evidence_chunk_id=getattr(
                result,
                "evidence_chunk_id",
                None,
            ),
        )
        for result in results
    ]


def _serialize_risk(state) -> RiskResponse | None:
    """Convert the risk assessment into an API response."""

    risk = _get_state_value(
        state,
        "risk_assessment",
        None,
    )

    if risk is None:
        return None

    return RiskResponse(
        risk_score=risk.risk_score,
        risk_level=risk.risk_level,
        triggers=list(risk.triggers),
        recommended_action=risk.recommended_action,
    )


def _audit_to_response(record: AuditRecord) -> AuditResponse:
    """Convert an AuditRecord domain object to an API response."""

    return AuditResponse(
        audit_id=record.audit_id,
        timestamp=record.timestamp,
        user_query=record.user_query,
        claim=record.claim,
        verification_status=record.verification_status,
        verification_reason=record.verification_reason,
        risk_level=record.risk_level,
        evidence=record.evidence,
        provenance=record.provenance,
        claim_id=record.claim_id,
        document_id=record.document_id,
        document_sha256=record.document_sha256,
        page_number=record.page_number,
        risk_assessment=record.risk_assessment,
        created_at=record.created_at,
        reviewer=record.reviewer,
        review_decision=record.review_decision,
        review_notes=record.review_notes,
        review_timestamp=record.review_timestamp,
        status=record.status,
        confidence_score=record.confidence_score,
        risk_score=record.risk_score,
        triggers=record.triggers,
        verification_results=record.verification_results,
    )


# ============================================================================
# System endpoints
# ============================================================================

@router.get(
    "/health",
    tags=["system"],
)
def health_check() -> dict[str, str]:
    """Return API health status."""

    return {
        "status": "ok",
    }


# ============================================================================
# Document endpoints
# ============================================================================

@router.get(
    "/documents",
    response_model=list[DocumentResponse],
    tags=["documents"],
)
def list_documents() -> List[DocumentResponse]:
    """
    List all available financial documents.
    
    Returns a list of all PDF documents found in the data directories.
    """
    pdf_files = _find_pdf_documents()
    
    documents = []
    for pdf in pdf_files:
        # Detect company from filename
        company = _detect_company_from_text(pdf.name)
        
        documents.append(
            DocumentResponse(
                name=pdf.name,
                path=str(pdf),
                size=pdf.stat().st_size,
                modified=pdf.stat().st_mtime,
                directory=str(pdf.parent),
                company=company,
            )
        )
    
    return sorted(documents, key=lambda x: x.name)


@router.get(
    "/documents/{document_name}",
    response_model=DocumentResponse,
    tags=["documents"],
)
def get_document(document_name: str) -> DocumentResponse:
    """
    Get metadata for a specific document.
    """
    pdf_files = _find_pdf_documents()
    
    for pdf in pdf_files:
        if pdf.name == document_name:
            company = _detect_company_from_text(pdf.name)
            return DocumentResponse(
                name=pdf.name,
                path=str(pdf),
                size=pdf.stat().st_size,
                modified=pdf.stat().st_mtime,
                directory=str(pdf.parent),
                company=company,
            )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Document '{document_name}' not found.",
    )


# ============================================================================
# Company endpoints
# ============================================================================

@router.get(
    "/companies",
    tags=["companies"],
)
def list_companies() -> dict:
    """
    List all detected companies and their documents.
    """
    pdf_files = _find_pdf_documents()
    
    companies = {}
    for pdf in pdf_files:
        company = _detect_company_from_text(pdf.name)
        if company:
            if company not in companies:
                companies[company] = []
            companies[company].append({
                "name": pdf.name,
                "path": str(pdf),
                "size": pdf.stat().st_size,
            })
    
    return {
        "companies": list(companies.keys()),
        "details": companies,
    }


@router.get(
    "/companies/{company_name}/documents",
    tags=["companies"],
)
def get_company_documents(company_name: str) -> List[DocumentResponse]:
    """
    Get all documents for a specific company.
    """
    pdf_files = _find_pdf_documents()
    
    company_docs = []
    for pdf in pdf_files:
        detected = _detect_company_from_text(pdf.name)
        if detected and detected.lower() == company_name.lower():
            company_docs.append(
                DocumentResponse(
                    name=pdf.name,
                    path=str(pdf),
                    size=pdf.stat().st_size,
                    modified=pdf.stat().st_mtime,
                    directory=str(pdf.parent),
                    company=detected,
                )
            )
    
    if not company_docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No documents found for company '{company_name}'.",
        )
    
    return sorted(company_docs, key=lambda x: x.name)


# ============================================================================
# Agent endpoints
# ============================================================================

@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["agent"],
)
def query(request: QueryRequest) -> QueryResponse:
    """
    Submit a financial question to the compliant RAG agent.

    The complete workflow is executed through the existing agent graph.
    
    Supports:
    - Single metric queries: "What was Apple's revenue in 2025?"
    - Multi-metric queries: "What was revenue, net income, and gross profit?"
    - Narrative queries: "Why did Apple's revenue change in 2025?"
    - Company-specific queries: "What was Microsoft's net income?"
    - Multi-company queries: "Compare Apple and Microsoft revenue"
    """
    try:
        state = run_agent(request.user_query)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent execution failed.",
        ) from exc

    final_answer = _get_state_value(
        state,
        "final_answer",
        "",
    )

    # If final_answer is empty, use raw_llm_output
    if not final_answer:
        final_answer = _get_state_value(
            state,
            "raw_llm_output",
            "",
        )

    final_status = _get_state_value(
        state,
        "final_response_status",
        None,
    )

    should_route_to_audit = bool(
        _get_state_value(
            state,
            "should_route_to_audit",
            False,
        )
    )

    audit_record = _get_state_value(
        state,
        "audit_record",
        None,
    )

    audit_id = None

    if audit_record is not None:
        audit_id = getattr(
            audit_record,
            "audit_id",
            None,
        )

    # Get claims count
    claims = _get_state_value(
        state,
        "claims",
        [],
    )

    return QueryResponse(
        final_answer=final_answer,
        status=(
            final_status.value
            if hasattr(final_status, "value")
            else str(final_status)
            if final_status is not None
            else "UNKNOWN"
        ),
        should_route_to_audit=should_route_to_audit,
        audit_id=audit_id,
        evidence=_serialize_evidence(state),
        verification_results=_serialize_verification_results(state),
        risk_assessment=_serialize_risk(state),
        claims_count=len(claims),
        query_analysis=_get_state_value(state, "query_analysis", None),
    )


# ============================================================================
# Audit endpoints
# ============================================================================

@router.get(
    "/audits",
    response_model=list[AuditResponse],
    tags=["audit"],
)
def get_pending_audits() -> list[AuditResponse]:
    """Return audit cases currently waiting for human review."""

    records = _review_service.get_pending_reviews()

    return [
        _audit_to_response(record)
        for record in records
    ]


@router.get(
    "/audits/{audit_id}",
    response_model=AuditResponse,
    tags=["audit"],
)
def get_audit(audit_id: str) -> AuditResponse:
    """Return a specific audit record."""

    record = _review_service.queue.get_by_id(
        audit_id
    )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit record '{audit_id}' was not found.",
        )

    return _audit_to_response(record)


@router.post(
    "/audits/{audit_id}/decision",
    response_model=AuditDecisionResponse,
    tags=["audit"],
)
def submit_audit_decision(
    audit_id: str,
    request: AuditDecisionRequest,
) -> AuditDecisionResponse:
    """
    Submit a human review decision for an audit case.

    The ReviewService controls the state transition:
    PENDING -> IN_REVIEW -> RESOLVED.
    """

    record = _review_service.queue.get_by_id(
        audit_id
    )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit record '{audit_id}' was not found.",
        )

    success = _review_service.submit_review_decision(
        audit_id=audit_id,
        decision=request.decision,
        notes=request.notes or "",
        reviewer=request.reviewer,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Audit record '{audit_id}' could not be resolved. "
                "It may already be under review or resolved."
            ),
        )

    updated_record = _review_service.queue.get_by_id(
        audit_id
    )

    if updated_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit record '{audit_id}' was not found after update.",
        )

    return AuditDecisionResponse(
        audit_id=updated_record.audit_id,
        status=updated_record.status,
        review_decision=updated_record.review_decision,
        reviewer=updated_record.reviewer or request.reviewer,
        review_notes=updated_record.review_notes,
        review_timestamp=updated_record.review_timestamp,
    )