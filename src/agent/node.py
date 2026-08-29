"""Agent graph nodes for retrieval, generation, verification, guardrails, and audit."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.agent.state import AgentState, FinalResponseStatus
from src.audit.review_service import ReviewService
from src.guardrails.final_safety import FinalSafetyValidator
from src.guardrails.policies import GuardrailPolicies
from src.guardrails.runner import GuardrailRunner
from src.ingestion.pipeline import FinancialDocumentPipeline
from src.llm.client import LLMClient, get_llm_client
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import VectorStore
from src.verification.claim_verifier import ClaimVerifier
from src.verification.models import Claim, ClaimType, VerificationStatus

# Backward-compatible verifier name expected by existing agent tests.
Verifier = ClaimVerifier


# ---------------------------------------------------------------------------
# Retrieval construction
# ---------------------------------------------------------------------------


def _find_pdf_documents() -> list[Path]:
    """Find financial PDF documents available to the project."""

    candidates = (
        Path("data"),
        Path("data/raw"),
        Path("data/documents"),
        Path("documents"),
    )

    pdf_files: list[Path] = []

    for directory in candidates:
        if directory.is_dir():
            pdf_files.extend(directory.rglob("*.pdf"))

    return sorted(set(pdf_files))


def _load_document_chunks() -> list[Any]:
    """Run the existing ingestion pipeline and return searchable chunks."""

    pdf_files = _find_pdf_documents()

    if not pdf_files:
        raise RuntimeError(
            "No financial PDF documents were found. "
            "Place the source PDF under data/, data/raw/, "
            "data/documents/, or documents."
        )

    pipeline = FinancialDocumentPipeline()
    chunks: list[Any] = []

    for pdf_file in pdf_files:
        result = pipeline.process(pdf_file)
        chunks.extend(result.chunks)

    if not chunks:
        raise RuntimeError(
            "Document ingestion completed, but no searchable chunks "
            "were produced."
        )

    return chunks


def _build_hybrid_retriever() -> HybridRetriever:
    """Build the project's BM25 + Vector + RRF retriever."""

    chunks = _load_document_chunks()

    bm25_retriever = BM25Retriever(chunks)
    vector_store = VectorStore(chunks)

    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
    )


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------


def query_analysis_node(state: AgentState) -> AgentState:
    """Analyze the user query."""

    query = state.user_query.strip()
    lowered = query.lower()

    entities: list[str] = []

    if "revenue" in lowered:
        entities.append("revenue")

    if "profit" in lowered:
        entities.append("profit")

    if "income" in lowered:
        entities.append("income")

    if "assets" in lowered:
        entities.append("assets")

    if "liabilities" in lowered:
        entities.append("liabilities")

    years = re.findall(r"\b(?:19|20)\d{2}\b", query)

    state.query_analysis = {
        "entities": entities,
        "period": years[0] if years else None,
        "all_metrics": [],  # Will be populated for multi-query
    }

    return state


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> AgentState:
    """Perform hybrid BM25 + vector retrieval using RRF."""

    query = state.user_query.strip()

    if not query:
        state.retrieval_results = []
        return state

    try:
        retriever = _build_hybrid_retriever()
        results = retriever.retrieve(query, top_k=10)
        state.retrieval_results = list(results)
    except RuntimeError as e:
        # Handle case where no documents are found
        state.retrieval_results = []
        state.error = str(e)

    return state


# ---------------------------------------------------------------------------
# Claim generation
# ---------------------------------------------------------------------------


def _extract_company_name(combined_evidence: str) -> str | None:
    """Extract the company name from the evidence."""
    patterns = [
        r"THE\s+REAL\s+BROKERAGE\s+INC\.",
        r"The Real Brokerage Inc\.",
        r"Real Brokerage Inc\.",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc\.|Corp\.|Company|LLC))",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, combined_evidence, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def _format_financial_value(num_value: float, is_in_thousands: bool = False) -> tuple[str, str | None]:
    """
    Format a financial value with appropriate unit.
    
    Returns: (formatted_value, unit)
    """
    abs_value = abs(num_value)
    
    # Check if value is in thousands (from financial statements)
    if is_in_thousands:
        # Value is in thousands - convert appropriately
        if abs_value >= 1000000:
            # Billions (value in thousands / 1,000,000)
            billion_value = abs_value / 1000000
            formatted = f"${billion_value:.2f}B"
            unit = "billion"
        elif abs_value >= 1000:
            # Millions (value in thousands / 1000)
            million_value = abs_value / 1000
            formatted = f"${million_value:.2f}M"
            unit = "million"
        else:
            # Thousands
            formatted = f"${abs_value:,.0f}K"
            unit = "thousand"
    else:
        # Value is not explicitly in thousands
        if abs_value >= 1000000000:
            # Billions
            billion_value = abs_value / 1000000000
            formatted = f"${billion_value:.2f}B"
            unit = "billion"
        elif abs_value >= 1000000:
            # Millions
            million_value = abs_value / 1000000
            formatted = f"${million_value:.2f}M"
            unit = "million"
        elif abs_value >= 1000:
            # Thousands
            thousand_value = abs_value / 1000
            formatted = f"${thousand_value:.2f}K"
            unit = "thousand"
        else:
            formatted = f"${abs_value:.0f}"
            unit = None
    
    # Add negative sign if the original value was negative
    if num_value < 0:
        formatted = f"({formatted})"
    
    return formatted, unit


def _is_narrative_query(query: str) -> bool:
    """Check if the query is asking for narrative explanation."""
    narrative_keywords = [
        'why', 'how', 'what caused', 'what drove', 'what led to',
        'reason', 'explain', 'describe', 'impact', 'effect',
        'what new', 'what initiatives', 'what projects',
        'what strategic', 'what changes', 'what happened'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in narrative_keywords)


def _extract_narrative_answer(
    query: str,
    retrieval_results: list[Any],
    llm: LLMClient,
    claims: list[Claim] = None
) -> str:
    """Extract narrative answer from evidence using LLM."""
    
    # Build evidence with source labels
    evidence_parts = []
    for i, r in enumerate(retrieval_results[:5], 1):
        # Extract relevant sentences
        sentences = r.text.split('.')
        relevant = []
        for sent in sentences:
            sent_lower = sent.lower()
            # Include sentences with relevant keywords
            if any(term in sent_lower for term in [
                'loss', 'expense', 'cost', 'operating', 'revenue', 
                'profit', 'margin', 'growth', 'increase', 'decrease',
                'launch', 'initiative', 'project', 'strategic',
                'market', 'driver', 'factor', 'reason', 'cause'
            ]):
                relevant.append(sent.strip())
        
        if relevant:
            evidence_parts.append(
                f"[SOURCE {i}, PAGE {r.page_number}] " + '. '.join(relevant[:4]) + '.'
            )
    
    evidence_text = '\n\n'.join(evidence_parts)
    
    if not evidence_text:
        return "The evidence does not contain information about this topic."
    
    # Include claims context if available
    claims_context = ""
    if claims:
        claim_summary = "\n".join([f"- {c.subject}: {c.value}" for c in claims])
        claims_context = f"\n\nEXTRACTED FINANCIAL DATA:\n{claim_summary}\n"
    
    prompt = f"""You are a financial analyst. Answer the question using ONLY the evidence below.

QUESTION: {query}

EVIDENCE:
{evidence_text}
{claims_context}

CRITICAL RULES:
1. ONLY use information from the evidence above
2. Do NOT invent any numbers, dates, or facts that are not in the evidence
3. If the evidence does not contain the answer, say: "The evidence does not contain information about this."
4. Cite specific evidence sources (e.g., "According to SOURCE 1...")
5. Be specific, concise, and factual
6. If you mention a number, it MUST be from the evidence

ANSWER:"""
    
    return str(llm.generate(prompt)).strip()


def _extract_numeric_claims(
    state: AgentState,
    combined_evidence: str,
    query_lower: str,
    is_in_thousands: bool,
    company_name: str | None,
) -> list[Claim]:
    """Extract numeric claims from the evidence."""
    claims: list[Claim] = []

    # Determine the year
    year = "2025"
    if "2024" in query_lower:
        year = "2024"
    elif "2023" in query_lower:
        year = "2023"

    # Define patterns for different metrics with keywords for detection
    metric_configs = {
        "net loss": {
            "patterns": [
                r"Net\s+Loss\s*[:$]?\s*\$?\s*\(?\s*([\d,]+)\)?",
                r"Net\s+Loss\s+(?:for|attributable)\s+.*?[:$]?\s*\$?\s*\(?\s*([\d,]+)\)?",
                r"Net\s+loss\s+was\s*\$?\s*\(?\s*([\d,]+)\)?",
            ],
            "subject": "net loss",
            "keywords": ["net loss", "net income", "loss"],
        },
        "revenue": {
            "patterns": [
                r"Total\s+Revenue\s*[:$]?\s*\$?\s*([\d,]+)",
                r"Revenues?\s*[:$]?\s*\$?\s*([\d,]+)",
                r"Revenue\s*[:$]?\s*\$?\s*([\d,]+)",
                r"revenue\s+(?:grew|was|of|to)\s*\$?\s*([\d,]+)\s*(million|billion)",
            ],
            "subject": "revenue",
            "keywords": ["revenue", "revenues", "sales"],
        },
        "gross profit": {
            "patterns": [
                r"Gross\s+Profit\s*[:$]?\s*\$?\s*([\d,]+)",
                r"Gross\s+profit\s+was\s*\$?\s*([\d,]+)",
            ],
            "subject": "gross profit",
            "keywords": ["gross profit", "gross margin"],
        },
        "operating loss": {
            "patterns": [
                r"Operating\s+Loss\s*[:$]?\s*\$?\s*\(?\s*([\d,]+)\)?",
                r"Operating\s+loss\s+was\s*\$?\s*\(?\s*([\d,]+)\)?",
            ],
            "subject": "operating loss",
            "keywords": ["operating loss", "operating income"],
        },
        "cost of sales": {
            "patterns": [
                r"Cost\s+of\s+Sales\s*[:$]?\s*\$?\s*([\d,]+)",
                r"Cost\s+of\s+sales\s+was\s*\$?\s*([\d,]+)",
            ],
            "subject": "cost of sales",
            "keywords": ["cost of sales", "cost of goods", "cogs"],
        },
        "operating expenses": {
            "patterns": [
                r"Operating\s+Expenses\s*[:$]?\s*\$?\s*([\d,]+)",
                r"Operating\s+expenses\s+was\s*\$?\s*([\d,]+)",
            ],
            "subject": "operating expenses",
            "keywords": ["operating expenses", "opex"],
        },
        "ebitda": {
            "patterns": [
                r"EBITDA\s*[:$]?\s*\$?\s*\(?\s*([\d,]+)\)?",
                r"Adjusted\s+EBITDA\s*[:$]?\s*\$?\s*\(?\s*([\d,]+)\)?",
            ],
            "subject": "EBITDA",
            "keywords": ["ebitda", "adjusted ebitda"],
        },
    }

    # Determine which metrics to extract based on the query
    # Check for multiple metrics in the query
    requested_metrics = []
    for metric_name, config in metric_configs.items():
        if any(keyword in query_lower for keyword in config["keywords"]):
            requested_metrics.append(metric_name)

    # If no specific metric found, check for all metrics
    if not requested_metrics:
        # Try to extract all metrics
        for metric_name, config in metric_configs.items():
            value = None
            for pattern in config["patterns"]:
                match = re.search(pattern, combined_evidence, re.IGNORECASE)
                if match:
                    value = match.group(1).replace(",", "")
                    break
            
            if value:
                num_value = float(value)
                if re.search(rf"{config['subject']}.*?\(\s*\$?\s*{value}", combined_evidence, re.IGNORECASE):
                    num_value = -num_value
                
                formatted_value, unit = _format_financial_value(num_value, is_in_thousands)
                
                subject = config["subject"]
                if company_name:
                    subject = f"{company_name} {subject}"
                
                claims.append(
                    Claim(
                        claim_id=f"claim_{len(claims) + 1}",
                        claim_type=ClaimType.NUMERIC,
                        subject=subject,
                        value=formatted_value,
                        unit=unit,
                        period=year,
                        source_chunk_id=(
                            state.retrieval_results[0].chunk_id
                            if state.retrieval_results
                            else None
                        ),
                    )
                )
        return claims

    # Extract specific requested metrics
    for metric_name in requested_metrics:
        config = metric_configs.get(metric_name)
        if not config:
            continue
            
        value = None
        for pattern in config["patterns"]:
            match = re.search(pattern, combined_evidence, re.IGNORECASE)
            if match:
                value = match.group(1).replace(",", "")
                break

        if value:
            num_value = float(value)
            if re.search(rf"{config['subject']}.*?\(\s*\$?\s*{value}", combined_evidence, re.IGNORECASE):
                num_value = -num_value
            
            formatted_value, unit = _format_financial_value(num_value, is_in_thousands)
            
            subject = config["subject"]
            if company_name:
                subject = f"{company_name} {subject}"
            
            claims.append(
                Claim(
                    claim_id=f"claim_{len(claims) + 1}",
                    claim_type=ClaimType.NUMERIC,
                    subject=subject,
                    value=formatted_value,
                    unit=unit,
                    period=year,
                    source_chunk_id=(
                        state.retrieval_results[0].chunk_id
                        if state.retrieval_results
                        else None
                    ),
                )
            )

    return claims


def claim_generation_node(state: AgentState) -> AgentState:
    """Generate candidate claims from retrieved evidence."""
    
    if not state.retrieval_results:
        state.raw_llm_output = (
            "The retrieved evidence is insufficient to answer this question."
        )
        state.claims = []
        return state

    # Combine all evidence text
    combined_evidence = "\n".join([r.text for r in state.retrieval_results])

    # Extract company name if the query asks for it
    query_lower = state.user_query.lower()
    company_name = None
    if "company" in query_lower or "name" in query_lower:
        company_name = _extract_company_name(combined_evidence)

    # Check if the evidence indicates values are in thousands
    is_in_thousands = "in thousands" in combined_evidence.lower()

    # Extract numeric claims (for both narrative and non-narrative)
    claims = _extract_numeric_claims(
        state,
        combined_evidence,
        query_lower,
        is_in_thousands,
        company_name,
    )

    # Check if this is a narrative query (why, how, explain, etc.)
    is_narrative = _is_narrative_query(state.user_query)

    if is_narrative:
        # For narrative queries, generate explanation using claims context
        llm = get_llm_client()
        narrative_answer = _extract_narrative_answer(
            state.user_query,
            state.retrieval_results,
            llm,
            claims  # Pass claims for context
        )
        state.raw_llm_output = narrative_answer
        state.claims = claims
        return state

    # For non-narrative queries, set raw output based on claims found
    if claims:
        # Build a summary from the claims
        claim_summaries = []
        for claim in claims:
            claim_summaries.append(f"{claim.subject} of {claim.value}")
        
        # If multiple claims, format as a list
        if len(claim_summaries) > 1:
            # Separate with commas and add "and" before the last
            formatted_summary = ", ".join(claim_summaries[:-1])
            if len(claim_summaries) > 2:
                formatted_summary += f", and {claim_summaries[-1]}"
            else:
                formatted_summary += f" and {claim_summaries[-1]}"
            state.raw_llm_output = f"The company reported {formatted_summary} in 2025."
        else:
            state.raw_llm_output = f"The company reported {claim_summaries[0]} in 2025."
    else:
        state.raw_llm_output = (
            "The evidence does not contain the requested financial information."
        )
    
    state.claims = claims
    return state


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verification_node(state: AgentState) -> AgentState:
    """Run deterministic verification on all extracted claims."""

    verifier = Verifier()

    evidence_text = "\n\n".join(
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


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def guardrail_node(state: AgentState) -> AgentState:
    """Run all configured guardrails."""

    policies = GuardrailPolicies()
    runner = GuardrailRunner(policies)

    result = runner.run_full_pipeline(
        query=state.user_query,
        retrieval_results=state.retrieval_results,
        verification_results=state.verification_results,
        raw_llm_output=state.raw_llm_output,
    )

    state.guardrail_result = result
    state.risk_assessment = result.risk_assessment if result else None
    state.should_route_to_audit = result.should_route_to_audit if result else False

    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def routing_node(state: AgentState) -> AgentState:
    """Pass through because guardrail_node determines routing."""

    return state


# ---------------------------------------------------------------------------
# Final answer
# ---------------------------------------------------------------------------


def answer_generation_node(state: AgentState) -> AgentState:
    """Generate the final answer from verified evidence."""

    verified_results = [
        result
        for result in state.verification_results
        if result.status == VerificationStatus.VERIFIED
    ]

    if state.claims and not verified_results:
        state.final_answer = (
            "The available evidence could not be deterministically "
            "verified, so the response requires human review."
        )
        state.final_response_status = (
            FinalResponseStatus.ROUTED_TO_AUDIT
        )
        return state

    # If there are no claims, provide a clear message
    if not state.claims:
        state.final_answer = (
            "The evidence does not contain the requested financial information. "
            "Please try a different query or check if the data is available."
        )
        state.final_response_status = (
            FinalResponseStatus.GENERATED
        )
        return state

    evidence_text = "\n\n".join(
        result.text
        for result in state.retrieval_results
    )

    prompt = f"""
Answer the user's question using ONLY the supplied evidence.

User question:
{state.user_query}

Verified claims:
{verified_results}

Evidence:
{evidence_text}

Do not introduce information that is not contained in the evidence.
Preserve financial periods, units, and scope.
If the evidence is insufficient, explicitly say so.
""".strip()

    llm = get_llm_client()

    state.final_answer = str(
        llm.generate(prompt)
    ).strip()

    return state


# ---------------------------------------------------------------------------
# Output guard
# ---------------------------------------------------------------------------


def output_guard_node(state: AgentState) -> AgentState:
    """Validate the generated answer before returning it."""

    validator = FinalSafetyValidator()

    # Safely access confidence score
    confidence_score = 1.0
    if state.guardrail_result:
        if hasattr(state.guardrail_result, 'confidence_score'):
            confidence_obj = state.guardrail_result.confidence_score
            if confidence_obj and hasattr(confidence_obj, 'overall'):
                confidence_score = confidence_obj.overall

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

        state.final_response_status = (
            FinalResponseStatus.BLOCKED
        )

    else:
        state.final_response_status = (
            FinalResponseStatus.GENERATED
        )

    return state


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_node(state: AgentState) -> AgentState:
    """Route the request to human review and create an audit record."""

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

    state.final_response_status = (
        FinalResponseStatus.ROUTED_TO_AUDIT
    )

    return state