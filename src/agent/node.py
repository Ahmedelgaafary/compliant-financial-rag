# src/agent/node.py
import re

from src.agent.state import AgentState, FinalResponseStatus
from src.audit.review_service import ReviewService
from src.guardrails.final_safety import FinalSafetyValidator
from src.guardrails.policies import GuardrailPolicies
from src.guardrails.runner import GuardrailRunner
from src.llm.client import LLMClient
from src.retrieval.hybrid import HybridRetriever
from src.verification.claim_verifier import ClaimVerifier
from src.verification.models import Claim, ClaimType, VerificationStatus

# Backward-compatible verifier name expected by existing agent tests.
Verifier = ClaimVerifier


def query_analysis_node(state: AgentState) -> AgentState:
    """Analyze query to extract entities, time periods, etc."""
    # Placeholder - replace with a real NER / LLM call.
    state.query_analysis = {
        "entities": ["revenue"],
        "period": "2025",
    }
    return state


def retrieval_node(state: AgentState) -> AgentState:
    """Perform hybrid retrieval (BM25 + Vector + RRF)."""
    retriever = HybridRetriever()
    results = retriever.retrieve(
        state.user_query,
        top_k=5,
    )
    state.retrieval_results = list(results)
    return state


def claim_generation_node(state: AgentState) -> AgentState:
    """Generate candidate claims from retrieved evidence."""
    llm = LLMClient()

    evidence_lines = [
        f"- {result.text}"
        for result in state.retrieval_results
    ]
    evidence_text = "\n".join(evidence_lines)

    prompt = (
        "Based on the evidence, generate a financial claim "
        "(e.g., revenue = $42.8B). Evidence:\n"
        f"{evidence_text}"
    )

    raw_output = llm.generate(prompt)
    state.raw_llm_output = raw_output

    match = re.search(
        r"(\w+)\s*=\s*\$?(\d+\.?\d*)\s*([BMK]?)",
        raw_output,
        re.IGNORECASE,
    )

    claims: list[Claim] = []

    if match:
        subject = match.group(1)
        value = f"${match.group(2)}"

        unit_map = {
            "B": "billion",
            "M": "million",
            "K": "thousand",
        }

        unit = unit_map.get(
            match.group(3).upper(),
            "",
        ) or None

        claim = Claim(
            claim_id=f"claim_{len(claims) + 1}",
            claim_type=ClaimType.NUMERIC,
            subject=subject,
            value=value,
            unit=unit,
            period=None,
            source_chunk_id=(
                state.retrieval_results[0].chunk_id
                if state.retrieval_results
                else None
            ),
        )

        claims.append(claim)

    state.claims = claims
    return state


def verification_node(state: AgentState) -> AgentState:
    """Run deterministic verification on all extracted claims."""
    verifier = Verifier()

    evidence_text = "\n".join(
        result.text
        for result in state.retrieval_results
    )

    verification_results = []

    for claim in state.claims:
        result = verifier.verify(
            claim,
            evidence_text,
        )
        verification_results.append(result)

    state.verification_results = verification_results
    return state


def guardrail_node(state: AgentState) -> AgentState:
    """Run all guardrails."""
    policies = GuardrailPolicies()
    runner = GuardrailRunner(policies)

    result = runner.run_full_pipeline(
        query=state.user_query,
        retrieval_results=state.retrieval_results,
        verification_results=state.verification_results,
        raw_llm_output=state.raw_llm_output,
    )

    state.guardrail_result = result
    state.risk_assessment = result.risk_assessment
    state.should_route_to_audit = result.should_route_to_audit

    return state


def routing_node(state: AgentState) -> AgentState:
    """Pass through - routing is decided by the guardrail result."""
    return state


def answer_generation_node(state: AgentState) -> AgentState:
    """Generate final answer using verified claims and evidence."""
    verified_claims = [
        result
        for result in state.verification_results
        if result.status == VerificationStatus.VERIFIED
    ]

    evidence_text = "\n".join(
        result.text
        for result in state.retrieval_results
    )

    prompt = (
        "Using only the following verified claims and evidence, "
        "answer the user's question.\n"
        f"Verified claims: {verified_claims}\n"
        f"Evidence: {evidence_text}\n"
        f"User query: {state.user_query}"
    )

    llm = LLMClient()
    state.final_answer = llm.generate(prompt)

    return state


def output_guard_node(state: AgentState) -> AgentState:
    """Validate the generated answer before returning it."""
    validator = FinalSafetyValidator()

    confidence_score = (
        state.guardrail_result.confidence_score.overall
        if (
            state.guardrail_result
            and state.guardrail_result.confidence_score
        )
        else 1.0
    )

    result = validator.validate(
        generated_answer=state.final_answer,
        verification_results=state.verification_results,
        retrieval_results=state.retrieval_results,
        risk_assessment=state.risk_assessment,
        confidence_score=confidence_score,
    )

    if not result.allowed:
        state.final_answer = (
            "This response could not be validated for safety. "
            "Please consult the original source or contact support."
        )
        state.error = "; ".join(result.reasons)
        state.final_response_status = FinalResponseStatus.BLOCKED
    else:
        state.final_response_status = FinalResponseStatus.GENERATED

    return state


def audit_node(state: AgentState) -> AgentState:
    """Route to human review and create an audit record."""
    review_service = ReviewService()

    outcome = review_service.initiate_review(
        user_query=state.user_query,
        claim=state.raw_llm_output or "No claim extracted",
        verification_status=(
            state.verification_results[0].status.value
            if state.verification_results
            else "inconclusive"
        ),
        verification_reason=(
            state.verification_results[0].reason
            if state.verification_results
            else "EVIDENCE_MISSING"
        ),
        risk_assessment=state.risk_assessment,
        verification_results=state.verification_results,
        evidence=[
            {
                "text": result.text,
                "page": result.page_number,
                "chunk_id": result.chunk_id,
            }
            for result in state.retrieval_results
        ],
        document_id=(
            state.retrieval_results[0].document_id
            if state.retrieval_results
            else ""
        ),
        document_sha256=(
            state.retrieval_results[0].document_sha256
            if state.retrieval_results
            else ""
        ),
        page_number=(
            state.retrieval_results[0].page_number
            if state.retrieval_results
            else 1
        ),
    )

    state.audit_record = outcome.audit_record

    state.final_answer = (
        "Your request has been sent for human review. "
        "You will be notified when a decision is made."
    )

    state.final_response_status = FinalResponseStatus.ROUTED_TO_AUDIT

    return state