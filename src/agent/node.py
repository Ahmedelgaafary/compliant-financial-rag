
"""Agent workflow nodes for retrieval, verification, guardrails, and audit."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.agent.query_analyzer import analyze_query, detect_companies
from src.agent.state import AgentState, AuditStatus, FinalResponseStatus
from src.audit.review_service import ReviewService
from src.guardrails.final_safety import FinalSafetyValidator
from src.guardrails.policies import GuardrailPolicies
from src.guardrails.runner import GuardrailRunner
from src.ingestion.pipeline import FinancialDocumentPipeline
from src.llm.client import LLMClient
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import RetrievalResult
from src.retrieval.vector_store import VectorStore
from src.verification.claim_verifier import ClaimVerifier
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)

Verifier = ClaimVerifier


# ----------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------


def get_llm_client() -> LLMClient:
    """Return the configured LLM client."""
    return LLMClient()


# ----------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------


def _detect_company_from_text(text: str) -> str | None:
    """Return the first detected company."""
    companies = detect_companies(text)
    return companies[0] if companies else None


def _find_pdf_documents() -> list[Path]:
    """Find financial PDFs in supported directories."""
    roots = (
        Path("data"),
        Path("data/raw"),
        Path("data/documents"),
        Path("documents"),
    )

    files: set[Path] = set()

    for root in roots:
        if root.is_dir():
            files.update(root.rglob("*.pdf"))

    return sorted(files)


def _load_document_chunks() -> list[Any]:
    """Load searchable chunks from all available financial PDFs."""
    files = _find_pdf_documents()

    if not files:
        raise RuntimeError("No financial PDF documents were found.")

    pipeline = FinancialDocumentPipeline()
    chunks: list[Any] = []

    for pdf_file in files:
        chunks.extend(pipeline.process(pdf_file).chunks)

    if not chunks:
        raise RuntimeError(
            "Document ingestion produced no searchable chunks."
        )

    return chunks


# ----------------------------------------------------------------------
# Query analysis
# ----------------------------------------------------------------------


def query_analysis_node(state: AgentState) -> AgentState:
    """Analyze the user's financial question."""
    query = state.user_query.strip()

    analysis = analyze_query(query)

    state.query_analysis = analysis
    state.query_tasks = list(analysis.get("tasks", []))

    return state


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------


def retrieval_node(state: AgentState) -> AgentState:
    """Retrieve evidence for each query task."""
    query = state.user_query.strip()

    if not query:
        state.retrieval_results = []
        state.retrieval_by_task = {}
        return state

    try:
        chunks = _load_document_chunks()

        retriever = HybridRetriever(
            bm25_retriever=BM25Retriever(chunks),
            vector_store=VectorStore(chunks),
        )

        analysis = analyze_query(query)

        state.query_analysis = analysis
        state.query_tasks = list(analysis.get("tasks", []))

        if not state.query_tasks:
            state.query_tasks = [
                {
                    "question_id": "q1",
                    "question": query,
                    "companies": [],
                    "metrics": [],
                    "period": _first_year(query),
                }
            ]

        grouped: dict[str, list[RetrievalResult]] = {}
        all_results: list[RetrievalResult] = []

        for task in state.query_tasks:
            question_id = task.get("question_id", "q1")
            question = task.get("question", query)
            companies = task.get("companies", [])

            for company in companies or [None]:
                retrieval_query = (
                    f"{company}: {question}"
                    if company
                    else question
                )

                results = list(
                    retriever.retrieve(
                        retrieval_query,
                        top_k=10,
                    )
                )

                if company:
                    scoped = [
                        result
                        for result in results
                        if _result_matches_company(result, company)
                    ]

                    if scoped:
                        results = scoped

                grouped.setdefault(question_id, []).extend(results)
                all_results.extend(results)

        state.retrieval_by_task = {
            key: _deduplicate_results(value)
            for key, value in grouped.items()
        }

        state.retrieval_results = _deduplicate_results(all_results)

    except RuntimeError as exc:
        state.retrieval_results = []
        state.retrieval_by_task = {}
        state.error = str(exc)

    return state


def _result_matches_company(
    result: RetrievalResult,
    company: str,
) -> bool:
    """Check whether retrieved evidence belongs to a company."""
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", company.lower())
        if len(token) > 2
    }

    haystack = (
        f"{result.document_id} {result.text[:2000]}"
    ).lower()

    return bool(tokens) and all(
        token in haystack for token in tokens
    )


def _deduplicate_results(
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Remove duplicate chunks while preserving highest scores."""
    seen: set[str] = set()
    output: list[RetrievalResult] = []

    for result in sorted(
        results,
        key=lambda item: (-item.score, item.chunk_id),
    ):
        if result.chunk_id in seen:
            continue

        seen.add(result.chunk_id)
        output.append(result)

    return output


# ----------------------------------------------------------------------
# Deterministic claim extraction
# ----------------------------------------------------------------------


_METRIC_PATTERNS = {
    "revenue": r"(?:total\s+)?(?:revenue|revenues|net\s+sales|sales)",
    "profit": r"(?:gross\s+profit|operating\s+profit|profit)",
    "income": r"(?:net\s+income|net\s+earnings|income)",
    "loss": r"(?:net\s+loss|operating\s+loss|loss)",
    "assets": r"(?:total\s+assets|current\s+assets|assets)",
    "liabilities": (
        r"(?:total\s+liabilities|current\s+liabilities|liabilities)"
    ),
    "cash flow": r"(?:operating\s+cash\s+flow|cash\s+flow)",
    "r&d": r"(?:research\s+and\s+development|r&d)",
    "ebitda": r"ebitda",
    "margin": (
        r"(?:gross\s+margin|operating\s+margin|net\s+margin|margin)"
    ),
    "growth": r"(?:growth|increase|decrease|change)",
}


_VALUE_PATTERN = (
    r"(?P<value>"
    r"\(?\s*\$?\s*[-+]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*(?:bn|mn|b|m|k|billion|million|thousand|trillion))?"
    r"\s*\)?"
    r")"
)


def claim_generation_node(state: AgentState) -> AgentState:
    """
    Extract deterministic financial claims from retrieved evidence.

    Numeric financial claims are extracted directly from retrieved
    evidence. The LLM is intentionally not used for numeric claim
    extraction.
    """
    claims: list[Claim] = []

    if not state.query_tasks:
        claims.extend(
            _extract_numeric_claims(
                results=state.retrieval_results,
                period=_first_year(state.user_query),
                question_id="q1",
            )
        )

    else:
        for task in state.query_tasks:
            question_id = task.get("question_id", "q1")
            question = task.get(
                "question",
                state.user_query,
            )

            companies = task.get("companies", [])
            metrics = task.get("metrics", []) or _infer_metrics(question)
            period = task.get("period")

            results = state.retrieval_by_task.get(
                question_id,
                state.retrieval_results,
            )

            for company in companies or [None]:
                scoped_results = results

                if company:
                    company_results = [
                        result
                        for result in results
                        if _result_matches_company(result, company)
                    ]

                    if company_results:
                        scoped_results = company_results

                for metric in metrics:
                    claims.extend(
                        _extract_metric_claims(
                            metric=metric,
                            results=scoped_results,
                            period=period,
                            company=company,
                            question_id=question_id,
                        )
                    )

    state.claims = _deduplicate_claims(claims)

    if state.claims:
        state.raw_llm_output = ""

    else:
        state.raw_llm_output = (
            "The evidence does not contain "
            "the requested financial information."
        )

    return state


def _extract_numeric_claims(
    results: list[RetrievalResult],
    period: str | None = None,
    question_id: str = "q1",
    company: str | None = None,
) -> list[Claim]:
    """
    Backward-compatible numeric claim extractor.

    Simple financial questions default to revenue extraction.
    """
    return _extract_metric_claims(
        metric="revenue",
        results=results,
        period=period,
        company=company,
        question_id=question_id,
    )


def _infer_metrics(question: str) -> list[str]:
    """Infer supported financial metrics from a question."""
    text = question.lower()

    return [
        metric
        for metric, pattern in _METRIC_PATTERNS.items()
        if re.search(pattern, text)
    ]


def _extract_metric_claims(
    metric: str,
    results: list[RetrievalResult],
    period: str | None,
    company: str | None,
    question_id: str,
) -> list[Claim]:
    """
    Extract one deterministic claim for a financial metric.
    """
    metric_key = metric.lower().strip()
    pattern = _METRIC_PATTERNS.get(metric_key)

    if not pattern:
        return []

    # Currency-denominated metrics require a $ sign or explicit unit
    currency_metrics = {
        "revenue", "profit", "income", "loss", "assets", 
        "liabilities", "cash flow", "r&d", "ebitda"
    }
    requires_currency = metric_key in currency_metrics

    patterns = (
        re.compile(
            rf"{pattern}"
            rf"\s*(?:was|were|is|are|of|to|=|:)?\s*"
            rf"{_VALUE_PATTERN}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"{_VALUE_PATTERN}"
            rf"\s+(?:in|for)\s+"
            rf"{pattern}",
            re.IGNORECASE,
        ),
    )

    all_matches = []

    for result in results:
        candidates = (
            _period_scoped_texts(
                result.text,
                period,
            )
            or [result.text]
        )

        for text in candidates:
            for regex in patterns:
                match = regex.search(text)

                if not match:
                    continue

                raw_value = match.group("value")
                context = text[max(0, match.start() - 50):min(len(text), match.end() + 50)]

                # ============================================================
                # SKIP PERCENTAGES
                # ============================================================
                if '%' in context or 'percent' in context.lower() or 'percentage' in context.lower():
                    continue

                if 'percentage of' in context.lower():
                    continue

                # Skip if the match is "14 %" or similar
                if re.search(rf"{_VALUE_PATTERN}\s*%", text[max(0, match.start() - 5):min(len(text), match.end() + 5)]):
                    continue

                numeric_part = re.sub(r'[^0-9.]', '', raw_value)
                if not numeric_part:
                    continue

                try:
                    num_val = float(numeric_part)
                except ValueError:
                    continue

                # Skip small numbers that are likely percentages
                if num_val < 100 and ('%' in context or 'percent' in context.lower()):
                    continue

                # ============================================================
                # REQUIRE $ SIGN OR UNIT FOR CURRENCY METRICS (CRITICAL FIX)
                # ============================================================
                has_dollar = '$' in raw_value or '$' in context
                unit = _extract_unit(raw_value)
                has_unit = unit is not None

                # For currency metrics, require a $ sign or a unit
                if requires_currency and not has_dollar and not has_unit:
                    # Check if the context has a $ sign nearby
                    if '$' not in context:
                        # Skip this match - it's likely a percentage or ratio
                        continue

                # For non-currency metrics (margin, growth), allow bare numbers
                # But still check if it's a percentage context
                if not requires_currency and '%' in context:
                    continue

                # ============================================================
                # Check if this is a "Total" match (prefer these)
                # ============================================================
                is_total = 'total net sales' in text.lower() or 'Total net sales' in text

                all_matches.append({
                    'result': result,
                    'text': text,
                    'match': match,
                    'raw_value': raw_value,
                    'numeric_value': num_val,
                    'unit': unit,
                    'has_unit': has_unit,
                    'has_dollar': has_dollar,
                    'is_total': is_total,
                    'context': context,
                })

    # ============================================================
    # PICK THE BEST MATCH
    # ============================================================
    if not all_matches:
        return []

    # Sort by priority:
    # 1. "Total net sales" matches first
    # 2. Has $ sign
    # 3. Has unit (billion, million, etc.)
    # 4. Larger numbers first
    all_matches.sort(
        key=lambda x: (
            # Priority 1: Total net sales
            not x['is_total'],
            # Priority 2: Has $ sign
            not x['has_dollar'],
            # Priority 3: Has unit
            not x['has_unit'],
            # Priority 4: Larger number (descending)
            -x['numeric_value'],
        )
    )

    # Take the best match
    best = all_matches[0]

    raw_value = best['raw_value']
    unit = best['unit']

    # If no unit was extracted, infer from context
    if unit is None:
        if "in millions" in best['text'].lower():
            unit = "million"
        elif best['numeric_value'] > 1000:
            if "Apple" in best['text'] or "Total net sales" in best['text']:
                unit = "million"
            else:
                unit = "thousand"

    normalized_value = _normalize_claim_value(raw_value)

    if not normalized_value:
        return []

    claim = Claim(
        claim_id=(
            f"{question_id}_claim_"
            f"{metric_key}_{best['result'].chunk_id}"
        ),
        claim_type=ClaimType.NUMERIC,
        subject=metric_key,
        value=normalized_value,
        unit=unit,
        period=period,
        source_chunk_id=best['result'].chunk_id,
        company_name=company,
        question_id=question_id,
    )

    return [claim]

def _period_scoped_texts(
    text: str,
    period: str | None,
) -> list[str]:
    """Return text surrounding occurrences of the requested period."""
    if not period or period not in text:
        return []

    return [
        text[
            max(0, match.start() - 500) :
            min(len(text), match.end() + 1500)
        ]
        for match in re.finditer(
            re.escape(period),
            text,
        )
    ]


def _normalize_claim_value(value: str) -> str:
    """
    Normalize a financial value into a canonical representation.

    Examples:
        "$42.8 billion" -> "$42.8B"
        "$42.8 billion" -> "$42.8B"
        "$42.8 b"       -> "$42.8B"
        "$1.2 million"  -> "$1.2M"
        "$500 mn"       -> "$500M"
    """
    normalized = " ".join(value.strip().split())

    normalized = normalized.replace("$ ", "$")

    # Remove unnecessary spaces immediately inside parentheses.
    normalized = re.sub(
        r"\(\s*",
        "(",
        normalized,
    )
    normalized = re.sub(
        r"\s*\)",
        ")",
        normalized,
    )

    unit_map = {
        "trillion": "T",
        "billion": "B",
        "million": "M",
        "thousand": "K",
        "bn": "B",
        "mn": "M",
        "b": "B",
        "m": "M",
        "k": "K",
    }

    match = re.search(
        r"\s*(trillion|billion|million|thousand|bn|mn|b|m|k)\s*$",
        normalized,
        re.IGNORECASE,
    )

    if match:
        unit = unit_map[match.group(1).lower()]
        normalized = (
            normalized[: match.start()].rstrip()
            + unit
        )

    return normalized


def _extract_unit(value: str) -> str | None:
    """Extract the canonical financial unit."""
    match = re.search(
        r"(T|B|M|K)$",
        value.strip(),
        re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(1).upper()


def _first_year(text: str) -> str | None:
    """Return the first four-digit year found in text."""
    match = re.search(
        r"\b(?:19|20)\d{2}\b",
        text,
    )

    return match.group(0) if match else None


def _deduplicate_claims(
    claims: list[Claim],
) -> list[Claim]:
    """Remove duplicate claims while preserving order."""
    seen: set[tuple] = set()
    output: list[Claim] = []

    for claim in claims:
        key = (
            claim.question_id,
            claim.company_name,
            claim.subject,
            claim.value,
            claim.period,
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(claim)

    return output


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------


def verification_node(state: AgentState) -> AgentState:
    """Verify every claim against its source chunk."""
    verifier = Verifier()

    evidence = {
        result.chunk_id: result.text
        for result in state.retrieval_results
    }

    results: list[VerificationResult] = []
    grouped: dict[str, list[VerificationResult]] = {}

    for claim in state.claims:
        result = verifier.verify(
            claim,
            evidence.get(
                claim.source_chunk_id or "",
                "",
            ),
        )

        results.append(result)

        question_id = claim.question_id or "q1"

        grouped.setdefault(
            question_id,
            [],
        ).append(result)

    state.verification_results = results
    state.verification_by_task = grouped

    return state


# ----------------------------------------------------------------------
# Guardrails
# ----------------------------------------------------------------------


def guardrail_node(state: AgentState) -> AgentState:
    """Run the configured guardrail pipeline."""
    runner = GuardrailRunner(GuardrailPolicies())

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


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------


def routing_node(state: AgentState) -> AgentState:
    """Routing pass-through node."""
    return state


# ----------------------------------------------------------------------
# Answer generation
# ----------------------------------------------------------------------


def answer_generation_node(state: AgentState) -> AgentState:
    """
    Generate a natural-language answer using verified claims only.

    A claim is eligible for generation only when its corresponding
    verification result has VERIFIED status.
    """
    verified_results = [
        result
        for result in state.verification_results
        if result.status == VerificationStatus.VERIFIED
    ]

    # Claims exist, but none were verified.
    if state.claims and not verified_results:
        state.final_answer = (
            "The available evidence could not be "
            "deterministically verified, so the "
            "response requires human review."
        )

        state.final_response_status = (
            FinalResponseStatus.ROUTED_TO_AUDIT
        )

        return state

    # No claims means there is nothing safe to generate.
    if not state.claims:
        state.final_answer = (
            "The evidence does not contain the requested "
            "financial information."
        )

        state.final_response_status = FinalResponseStatus.GENERATED

        return state

    verified_ids = {
        result.claim_id
        for result in verified_results
    }

    verified_claims = [
        claim
        for claim in state.claims
        if claim.claim_id in verified_ids
    ]

    # Defensive check: a verification result should always correspond
    # to an actual claim.
    if not verified_claims:
        state.final_answer = (
            "The available evidence could not be "
            "deterministically verified, so the "
            "response requires human review."
        )

        state.final_response_status = (
            FinalResponseStatus.ROUTED_TO_AUDIT
        )

        return state

    evidence = {
        result.chunk_id: result
        for result in state.retrieval_results
    }

    blocks: list[str] = []

    for claim in verified_claims:
        result = evidence.get(
            claim.source_chunk_id or ""
        )

        if not result:
            continue

        blocks.append(
            f"Company: {claim.company_name or 'Not specified'}\n"
            f"Claim: {claim.subject} = {claim.value}\n"
            f"Period: {claim.period or 'Not specified'}\n"
            f"Document: {result.document_id}\n"
            f"Page: {result.page_number}\n"
            f"Evidence: {result.text}"
        )

    # A verified claim without source evidence should never be sent
    # to the answer generator.
    if not blocks:
        state.final_answer = (
            "The verified claim could not be linked to "
            "its source evidence, so the response requires "
            "human review."
        )

        state.final_response_status = (
            FinalResponseStatus.ROUTED_TO_AUDIT
        )

        return state

    evidence_text = "\n\n".join(blocks)

    prompt = f"""
Answer the user's financial question using only the verified claims
and evidence below.

Question:

{state.user_query}

Verified claims:

{verified_claims}

Evidence:

{evidence_text}

Rules:

- Keep companies separated.
- Do not invent values.
- Preserve periods and units.
- Do not transfer values between companies.
- State clearly when a requested item is unavailable.
""".strip()

    # Resolve the client at generation time. This keeps the dependency
    # injectable and allows tests to replace get_llm_client().
    llm = get_llm_client()

    generated = llm.generate(prompt)

    state.final_answer = str(generated).strip()
    state.final_response_status = FinalResponseStatus.GENERATED

    return state


# ----------------------------------------------------------------------
# Output safety
# ----------------------------------------------------------------------


def output_guard_node(state: AgentState) -> AgentState:
    """Validate the generated answer."""
    validator = FinalSafetyValidator()

    confidence = 1.0

    if state.guardrail_result:
        confidence_obj = getattr(
            state.guardrail_result,
            "confidence_score",
            None,
        )

        if confidence_obj is not None:
            confidence = getattr(
                confidence_obj,
                "overall",
                1.0,
            )

    result = validator.validate(
        generated_answer=state.final_answer,
        verification_results=state.verification_results,
        retrieval_results=state.retrieval_results,
        risk_assessment=state.risk_assessment,
        confidence_score=confidence,
    )

    if not result.allowed:
        state.final_answer = (
            "This response could not be validated for safety. "
            "Please consult the original source or contact support."
        )

        state.error = "; ".join(
            getattr(result, "reasons", [])
        )

        state.final_response_status = FinalResponseStatus.BLOCKED

    return state


# ----------------------------------------------------------------------
# Audit
# ----------------------------------------------------------------------


def audit_node(state: AgentState) -> AgentState:
    """Send the request to human review."""
    service = ReviewService()

    verification = (
        state.verification_results[0]
        if state.verification_results
        else None
    )

    retrieval = (
        state.retrieval_results[0]
        if state.retrieval_results
        else None
    )

    outcome = service.initiate_review(
        user_query=state.user_query,
        claim=(
            state.raw_llm_output
            or "Financial request requires human review"
        ),
        verification_status=(
            verification.status.value
            if verification
            else "inconclusive"
        ),
        verification_reason=(
            verification.reason
            if verification
            else "EVIDENCE_MISSING"
        ),
        risk_assessment=state.risk_assessment,
        verification_results=state.verification_results,
        evidence=[
            {
                "text": result.text,
                "page": result.page_number,
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
            }
            for result in state.retrieval_results
        ],
        document_id=(
            retrieval.document_id
            if retrieval
            else ""
        ),
        document_sha256=(
            retrieval.document_sha256
            if retrieval
            else ""
        ),
        page_number=(
            retrieval.page_number
            if retrieval
            else 1
        ),
    )

    state.audit_record = outcome.audit_record
    state.audit_status = AuditStatus.ROUTED

    state.final_answer = (
        "Your request has been sent for human review. "
        "You will be notified when a decision is made."
    )

    state.final_response_status = (
        FinalResponseStatus.ROUTED_TO_AUDIT
    )

    return state


# ----------------------------------------------------------------------
# Compatibility
# ----------------------------------------------------------------------


def financial_intelligence_node(
    state: AgentState,
) -> AgentState:
    """Compatibility node for the financial-intelligence stage."""
    return state

