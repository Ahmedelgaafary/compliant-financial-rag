from dataclasses import dataclass, field
from typing import List, Optional

from src.audit.models import AuditRecord
from src.guardrails.risk_engine import RiskAssessment
from src.guardrails.runner import GuardrailPipelineResult
from src.retrieval.models import RetrievalResult
from src.verification.models import Claim, VerificationResult


@dataclass
class AgentState:
    """State for the LangGraph workflow."""
    # Input
    user_query: str = ""
    query_analysis: Optional[dict] = None

    # Retrieval
    retrieval_results: List[RetrievalResult] = field(default_factory=list)

    # Claims and verification
    raw_llm_output: Optional[str] = None
    claims: List[Claim] = field(default_factory=list)
    verification_results: List[VerificationResult] = field(default_factory=list)

    # Guardrails & risk
    guardrail_result: Optional[GuardrailPipelineResult] = None
    risk_assessment: Optional[RiskAssessment] = None

    # Audit
    audit_record: Optional[AuditRecord] = None
    audit_decision: Optional[str] = None

    # Final output
    final_answer: str = ""
    should_route_to_audit: bool = False
    error: Optional[str] = None