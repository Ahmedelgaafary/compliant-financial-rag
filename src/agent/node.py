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
from src.retrieval.retrieval_pipeline import RetrievalPipeline
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
    """Retrieve evidence for each query task using multi-stage pipeline."""
    query = state.user_query.strip()

    if not query:
        state.retrieval_results = []
        state.retrieval_by_task = {}
        return state

    try:
        chunks = _load_document_chunks()

        hybrid_retriever = HybridRetriever(
            bm25_retriever=BM25Retriever(chunks),
            vector_store=VectorStore(chunks),
        )

        pipeline = RetrievalPipeline(
            hybrid_retriever=hybrid_retriever,
            chunks=chunks,
        )

        analysis = analyze_query(query)
        state.query_analysis = analysis
        state.query_tasks = list(analysis.get("tasks", []))

        if not state.query_tasks:
            state.query_tasks = [
                {
                    "question_id": "q1",
                    "question": query,
                    "companies": detect_companies(query),
                    "period": _first_year(query),
                }
            ]

        grouped: dict[str, list[RetrievalResult]] = {}
        all_results: list[RetrievalResult] = []

        for task in state.query_tasks:
            question_id = task.get("question_id", "q1")
            question = task.get("question", query)
            companies = task.get("companies", [])
            period = task.get("period")
            metrics = task.get("metrics", [])

            if companies:
                for company in companies:
                    results = pipeline.retrieve(
                        query=question,
                        company=company,
                        metric=metrics[0] if metrics else None,
                        period=period,
                        # A larger top_k gives claim_generation_node's
                        # year-column alignment check more candidates
                        # to search through. The reranker's scoring
                        # can tie multiple chunks together (see
                        # FinancialReranker), so the correct evidence
                        # is not guaranteed to rank first - only that
                        # it's likely present somewhere in a wider
                        # candidate set. Extraction already rejects
                        # wrong-year matches and keeps searching, so
                        # this only helps it find the right one rather
                        # than risking it being excluded entirely.
                        top_k=25,
                    )

                    company_key = f"{question_id}_{company}"
                    grouped.setdefault(company_key, []).extend(results)
                    all_results.extend(results)
            else:
                results = pipeline.retrieve(
                    query=question,
                    period=period,
                    top_k=25,
                )
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
    """
    Check whether retrieved evidence belongs to a company.
    Uses document_id, text content, and company aliases.
    """
    from src.config.companies import CompanyConfig

    if not company:
        return True  # No company specified, accept all

    company_lower = company.lower().strip()

    # 1. Check if document_id directly matches
    if result.document_id:
        doc_lower = result.document_id.lower()

        if company_lower in doc_lower:
            return True

        if "apple" in doc_lower and ("apple" in company_lower or "aapl" in company_lower):
            return True

        if "msft" in doc_lower and ("microsoft" in company_lower or "msft" in company_lower):
            return True

        detected_key = CompanyConfig.detect_company(result.document_id)
        if detected_key:
            if detected_key.lower() in company_lower or company_lower in detected_key.lower():
                return True

        for key, config in CompanyConfig.COMPANIES.items():
            for doc_id_pattern in config.get("document_ids", []):
                if doc_id_pattern.lower() in doc_lower:
                    if key.lower() in company_lower or company_lower in key.lower():
                        return True

    # 2. Check text content
    if result.text:
        text_lower = result.text.lower()

        if company_lower in text_lower:
            return True

        config = CompanyConfig.get_company(company)
        if config:
            for variation in config.get("variations", []):
                if variation.lower() in text_lower:
                    return True

            for keyword in config.get("keywords", []):
                if keyword.lower() in text_lower:
                    return True

    return False


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
    "net_income": r"(?:net\s+income|net\s+earnings)",
    "loss": r"(?:net\s+loss|operating\s+loss|loss)",
    "assets": r"(?:total\s+assets|current\s+assets|assets)",
    "liabilities": r"(?:total\s+liabilities|current\s+liabilities|liabilities)",
    "cash flow": r"(?:operating\s+cash\s+flow|cash\s+flow)",
    "r&d": r"(?:research\s+and\s+development|r&d)",
    "ebitda": r"ebitda",
    "margin": r"(?:gross\s+margin|operating\s+margin|net\s+margin|margin)",
    "growth": r"(?:growth|increase|decrease|change)",
}


_PERCENTAGE_METRICS = {"margin", "growth"}


_VALUE_PATTERN = (
    r"(?P<value>"
    r"\(?\s*\$?\s*[-+]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*(?:bn|mn|b|m|k|billion|million|thousand|trillion))?"
    r"\s*\)?"
    r")"
)


_YEAR_TOKEN = r"(?:19|20)\d{2}"

_YEAR_HEADER_PATTERN = re.compile(
    rf"(?:{_YEAR_TOKEN}(?:[\s,]+(?:and\s+)?)){{1,3}}{_YEAR_TOKEN}"
)

_CURRENCY_VALUE_PATTERN = re.compile(
    r"\$\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*(?:bn|mn|b|m|k|billion|million|thousand|trillion))?",
    re.IGNORECASE,
)


def _nearest_year_header(
    text: str,
    end_pos: int,
    window: int = 500,
) -> list[str]:
    """Find the closest run of 2-4 years appearing before end_pos."""
    start = max(0, end_pos - window)
    segment = text[start:end_pos]

    header_matches = list(_YEAR_HEADER_PATTERN.finditer(segment))

    if not header_matches:
        return []

    closest = header_matches[-1]

    return re.findall(_YEAR_TOKEN, closest.group(0))


def _values_after(
    text: str,
    start_pos: int,
    count: int,
    window: int = 150,
) -> list[str]:
    """Collect up to count consecutive currency values starting near start_pos."""
    end = min(len(text), start_pos + window)
    segment = text[start_pos:end]

    return [
        m.group(0)
        for m in _CURRENCY_VALUE_PATTERN.finditer(segment)
    ][:count]


_STANDALONE_YEAR_PATTERN = re.compile(rf"\b{_YEAR_TOKEN}\b")


def _nearest_standalone_year(
    text: str,
    end_pos: int,
    window: int = 300,
) -> str | None:
    """
    Find the closest single standalone year appearing shortly before
    end_pos, for use when no multi-year table header run was found.
    See _resolve_value_by_year_column for why this matters.
    """
    start = max(0, end_pos - window)
    segment = text[start:end_pos]

    matches = list(_STANDALONE_YEAR_PATTERN.finditer(segment))

    if not matches:
        return None

    return matches[-1].group(0)


def _resolve_value_by_year_column(
    text: str,
    match: re.Match,
    period: str | None,
) -> tuple[str | None, bool]:
    """
    Align a matched metric value to the requested fiscal year.

    Returns (value, is_mismatch):
    - (None, False): no discernible year context nearby - the caller
      should fall back to its normal single-value match.
    - (None, True): a year context WAS found (either a multi-year
      table header, or a standalone single-year section marker), but
      it does not match the requested period - the located figures
      clearly belong to a different fiscal year, so the caller should
      reject this match outright rather than guessing.
    - (value, False): a multi-year header includes the requested
      period, and `value` is the correctly column-aligned figure.
    """
    years = _nearest_year_header(text, match.start())

    if len(years) >= 2:
        if not period or period not in years:
            return None, True

        values = _values_after(text, match.start(), len(years))

        if len(values) < len(years):
            return None, True

        return values[years.index(period)], False

    # No multi-year table header nearby. Some filings present each
    # fiscal year as its own separate mini-table instead (e.g. a
    # segment note reading "2024 Americas Europe ... Net sales $X"
    # followed later by an entirely separate "2023 Americas Europe
    # ... Net sales $Y" block) - _nearest_year_header only recognizes
    # runs of 2+ years, so it can't see this. Without this fallback
    # check, a page footer citing the filing's cover year (e.g.
    # "Apple Inc. | 2025 Form 10-K | 47") can be the ONLY occurrence
    # of the requested period anywhere in the chunk, causing whichever
    # unrelated year's standalone mini-table happens to be physically
    # closest to that footer to be scoped in and mistaken for the
    # requested year's data.
    standalone_year = _nearest_standalone_year(text, match.start())

    if standalone_year and period and standalone_year != period:
        return None, True

    return None, False


_CONTEXT_UNIT_PATTERN = re.compile(
    r"\(\s*in\s+(thousands?|millions?|billions?|trillions?)"
    r"(?:\s+of\s+\w+)?\s*\)",
    re.IGNORECASE,
)


def _nearest_context_unit(
    text: str,
    end_pos: int,
    window: int = 500,
) -> str | None:
    """Find the closest '(in millions)'-style annotation before end_pos."""
    start = max(0, end_pos - window)
    segment = text[start:end_pos]

    matches = list(_CONTEXT_UNIT_PATTERN.finditer(segment))

    if not matches:
        return None

    unit = matches[-1].group(1).casefold()

    return unit[:-1] if unit.endswith("s") else unit


def claim_generation_node(state: AgentState) -> AgentState:
    """Extract deterministic financial claims from retrieved evidence."""
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
            question = task.get("question", state.user_query)

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
            "The evidence does not contain the requested financial information."
        )

    return state


def _extract_numeric_claims(
    results: list[RetrievalResult],
    period: str | None = None,
    question_id: str = "q1",
    company: str | None = None,
) -> list[Claim]:
    """Backward-compatible numeric claim extractor defaulting to revenue."""
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
    """Normalize a financial value into a canonical representation."""
    normalized = " ".join(value.strip().split())

    normalized = normalized.replace("$ ", "$")

    normalized = re.sub(r"\(\s*", "(", normalized)
    normalized = re.sub(r"\s*\)", ")", normalized)

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
        normalized = normalized[: match.start()].rstrip() + unit

    return normalized


def _extract_unit(value: str) -> str | None:
    """Extract the canonical financial unit."""
    match = re.search(r"(T|B|M|K)$", value.strip(), re.IGNORECASE)

    if not match:
        return None

    return match.group(1).upper()


_PREFER_TOTAL_METRICS = {"revenue"}

_STRICT_TOTAL_OVERRIDES = {
    "revenue": r"total\s+(?:revenue|revenues|net\s+sales|sales)",
}


def _extract_metric_claims(
    metric: str,
    results: list[RetrievalResult],
    period: str | None,
    company: str | None,
    question_id: str,
) -> list[Claim]:
    """
    Extract one deterministic claim for a metric.

    For metrics prone to segment/regional breakdown ambiguity (right
    now: revenue), a filing's consolidated total ("Total net sales
    $416,161") and a same-keyword segment row ("Americas net sales
    $162,560") are both genuinely correct dollar figures for the
    requested year - nothing about year-alignment or currency
    formatting distinguishes them. Search first for an explicit
    "Total ..." match across all evidence; only fall back to the
    looser optional-total pattern if no consolidated figure is found
    anywhere, since some filings may only report the total under
    different wording.
    """
    metric_key = metric.lower()
    pattern = _METRIC_PATTERNS.get(metric_key)

    if not pattern:
        return []

    if metric_key in _PREFER_TOTAL_METRICS:
        strict_claims = _search_metric_pattern(
            _STRICT_TOTAL_OVERRIDES[metric_key],
            metric_key,
            results,
            period,
            company,
            question_id,
        )

        if strict_claims:
            return strict_claims

    return _search_metric_pattern(
        pattern,
        metric_key,
        results,
        period,
        company,
        question_id,
    )


def _search_metric_pattern(
    pattern: str,
    metric_key: str,
    results: list[RetrievalResult],
    period: str | None,
    company: str | None,
    question_id: str,
) -> list[Claim]:
    """Search evidence for one metric-keyword pattern and return the first valid claim."""
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

    for result in results:
        candidates = (
            _period_scoped_texts(result.text, period)
            or [result.text]
        )

        for text in candidates:
            for regex in patterns:
                match = regex.search(text)

                if not match:
                    continue

                column_value, mismatch = _resolve_value_by_year_column(
                    text, match, period
                )

                if mismatch:
                    continue

                raw_value = column_value or match.group("value")

                value = _normalize_claim_value(raw_value)

                if not value:
                    continue

                if metric_key not in _PERCENTAGE_METRICS:
                    matched_text = column_value or match.group(0)

                    if (
                        "$" not in matched_text
                        and _extract_unit(value) is None
                    ):
                        continue

                extracted_unit = _extract_unit(value)

                claim_unit = extracted_unit or (
                    _nearest_context_unit(text, match.start())
                )

                return [
                    Claim(
                        claim_id=(
                            f"{question_id}_claim_"
                            f"{metric_key}_{result.chunk_id}"
                        ),
                        claim_type=ClaimType.NUMERIC,
                        subject=metric_key,
                        value=value,
                        unit=claim_unit,
                        period=period,
                        source_chunk_id=result.chunk_id,
                        company_name=company,
                        question_id=question_id,
                    )
                ]

    return []


def _first_year(text: str) -> str | None:
    """Return the first four-digit year found in text."""
    match = re.search(r"\b(?:19|20)\d{2}\b", text)

    return match.group(0) if match else None


def _deduplicate_claims(claims: list[Claim]) -> list[Claim]:
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
            evidence.get(claim.source_chunk_id or "", ""),
        )

        results.append(result)

        question_id = claim.question_id or "q1"

        grouped.setdefault(question_id, []).append(result)

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
    """Generate a natural-language answer using verified claims only."""
    verified_results = [
        result
        for result in state.verification_results
        if result.status == VerificationStatus.VERIFIED
    ]

    if state.claims and not verified_results:
        state.final_answer = (
            "The available evidence could not be "
            "deterministically verified, so the "
            "response requires human review."
        )
        state.final_response_status = FinalResponseStatus.ROUTED_TO_AUDIT
        return state

    if not state.claims:
        state.final_answer = (
            "The evidence does not contain the requested "
            "financial information."
        )
        state.final_response_status = FinalResponseStatus.GENERATED
        return state

    verified_ids = {result.claim_id for result in verified_results}

    verified_claims = [
        claim for claim in state.claims if claim.claim_id in verified_ids
    ]

    if not verified_claims:
        state.final_answer = (
            "The available evidence could not be "
            "deterministically verified, so the "
            "response requires human review."
        )
        state.final_response_status = FinalResponseStatus.ROUTED_TO_AUDIT
        return state

    evidence = {
        result.chunk_id: result for result in state.retrieval_results
    }

    blocks: list[str] = []

    for claim in verified_claims:
        result = evidence.get(claim.source_chunk_id or "")

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

    if not blocks:
        state.final_answer = (
            "The verified claim could not be linked to "
            "its source evidence, so the response requires "
            "human review."
        )
        state.final_response_status = FinalResponseStatus.ROUTED_TO_AUDIT
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
            state.guardrail_result, "confidence_score", None
        )

        if confidence_obj is not None:
            confidence = getattr(confidence_obj, "overall", 1.0)

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
        state.error = "; ".join(getattr(result, "reasons", []))
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
        state.retrieval_results[0] if state.retrieval_results else None
    )

    outcome = service.initiate_review(
        user_query=state.user_query,
        claim=(
            state.raw_llm_output
            or "Financial request requires human review"
        ),
        verification_status=(
            verification.status.value if verification else "inconclusive"
        ),
        verification_reason=(
            verification.reason if verification else "EVIDENCE_MISSING"
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
        document_id=retrieval.document_id if retrieval else "",
        document_sha256=retrieval.document_sha256 if retrieval else "",
        page_number=retrieval.page_number if retrieval else 1,
    )

    state.audit_record = outcome.audit_record
    state.audit_status = AuditStatus.ROUTED

    state.final_answer = (
        "Your request has been sent for human review. "
        "You will be notified when a decision is made."
    )

    state.final_response_status = FinalResponseStatus.ROUTED_TO_AUDIT

    return state


# ----------------------------------------------------------------------
# Compatibility
# ----------------------------------------------------------------------


def financial_intelligence_node(state: AgentState) -> AgentState:
    """Compatibility node for the financial-intelligence stage."""
    return state