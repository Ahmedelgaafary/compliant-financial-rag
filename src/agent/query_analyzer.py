"""Query analyzer for multi-company and multi-question financial queries."""

from __future__ import annotations

import re
from typing import Any

from src.config.companies import CompanyConfig

_METRIC_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "revenue",
        (
            "revenue",
            "revenues",
            "sales",
            "net sales",
            "turnover",
        ),
    ),
    (
        "profit",
        (
            "profit",
            "gross profit",
            "operating profit",
        ),
    ),
    (
        "margin",
        (
            "margin",
            "gross margin",
            "operating margin",
            "net margin",
        ),
    ),
    (
        "income",
        (
            "income",
            "net income",
            "earnings",
            "net earnings",
        ),
    ),
    (
        "loss",
        (
            "loss",
            "net loss",
            "operating loss",
        ),
    ),
    (
        "assets",
        (
            "assets",
            "total assets",
            "current assets",
        ),
    ),
    (
        "liabilities",
        (
            "liabilities",
            "total liabilities",
            "current liabilities",
        ),
    ),
    (
        "cash flow",
        (
            "cash flow",
            "operating cash flow",
            "cash",
        ),
    ),
    (
        "r&d",
        (
            "r&d",
            "research and development",
            "research",
            "development",
        ),
    ),
    (
        "ebitda",
        (
            "ebitda",
        ),
    ),
    (
        "growth",
        (
            "growth",
            "increase",
            "decrease",
            "change",
        ),
    ),
)

_NARRATIVE_KEYWORDS = (
    "why",
    "how",
    "what caused",
    "what drove",
    "what led to",
    "reason",
    "explain",
    "describe",
    "impact",
    "effect",
)

# Common aliases for documents
_DEFAULT_COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "Apple": (
        "apple",
        "apple inc",
        "apple inc.",
    ),
    "The Real Brokerage Inc.": (
        "real brokerage",
        "the real brokerage",
        "the real brokerage inc",
        "the real brokerage inc.",
        "brokerage",
    ),
}


def split_questions(user_query: str) -> list[str]:
    """
    Split a request into independent questions.
    
    Supports:
        What was revenue in 2025 and what were assets?
        What was Apple's revenue in 2025? What was its profit?
        Compare Apple and Brokerage revenue in 2025.
    """
    query = user_query.strip()
    if not query:
        return []

    # Explicit question marks are the strongest delimiter
    parts = [
        part.strip()
        for part in re.split(r"\?(?=\s|$)", query)
        if part.strip()
    ]

    if len(parts) > 1:
        return [
            _ensure_question_mark(part)
            for part in parts
        ]

    # Handle "and what / and why / and how / and did..." style queries
    conjunction_pattern = re.compile(
        r"\s+\b(?:and|also)\s+"
        r"(?=(?:what|why|how|which|where|when|did|does|"
        r"do|was|were|is|are|has|have|compare|calculate)\b)",
        re.IGNORECASE,
    )

    parts = [
        part.strip()
        for part in conjunction_pattern.split(query)
        if part.strip()
    ]

    return [
        _ensure_question_mark(part)
        for part in parts
    ]


def detect_companies(
    text: str,
    known_companies: list[str] | None = None,
) -> list[str]:
    """
    Detect ALL companies explicitly mentioned in text.
    Returns deduplicated list of canonical company names.
    """
    
    aliases = dict(_DEFAULT_COMPANY_ALIASES)
    
    # Add known companies from config
    for company in known_companies or []:
        normalized = company.strip()
        if normalized:
            aliases.setdefault(
                normalized,
                (normalized.casefold(),),
            )
    
    # Also check CompanyConfig for additional companies
    for company_key, config in CompanyConfig.COMPANIES.items():
        company_name = config.get("name", company_key)
        variations = config.get("variations", [])
        aliases.setdefault(
            company_name,
            tuple(variations + [company_key]),
        )
    
    found: list[tuple[int, str]] = []
    lowered = text.casefold()
    
    # Find ALL companies mentioned
    for canonical, company_aliases in aliases.items():
        # Check if any alias appears in the text
        for alias in company_aliases:
            alias_lower = alias.casefold()
            # Use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(alias_lower) + r'\b'
            if re.search(pattern, lowered):
                position = lowered.find(alias_lower)
                if position >= 0:
                    found.append((position, canonical))
                    break  # Found this company, move to next
    
    # Sort by position (first occurrence)
    found.sort(key=lambda item: item[0])
    
    # Return unique companies in order of appearance
    seen = set()
    result = []
    for _, company in found:
        # Normalize: if we have both "Apple" and "Apple Inc.", keep only one
        normalized_company = _normalize_company_name(company)
        if normalized_company not in seen:
            seen.add(normalized_company)
            result.append(normalized_company)
    
    return result


def _normalize_company_name(company: str) -> str:
    """Normalize company name to avoid duplicates."""
    company_lower = company.lower()
    
    # Map variations to canonical names
    variations = {
        "apple": "Apple",
        "apple inc.": "Apple",
        "apple inc": "Apple",
        "aapl": "Apple",
        
        "microsoft": "Microsoft",
        "microsoft corp.": "Microsoft",
        "microsoft corp": "Microsoft",
        "msft": "Microsoft",
        
        "google": "Google",
        "alphabet": "Google",
        "googl": "Google",
        
        "amazon": "Amazon",
        "amazon.com": "Amazon",
        "amzn": "Amazon",
        
        "real brokerage": "Real Brokerage",
        "the real brokerage": "Real Brokerage",
        "real": "Real Brokerage",
        
        "tesla": "Tesla",
        "tsla": "Tesla",
    }
    
    # Check if the company name is a variation of a known company
    if company_lower in variations:
        return variations[company_lower]
    
    # Check if any part of the name matches a variation
    for var, canonical in variations.items():
        if var in company_lower:
            return canonical
    
    return company


def detect_metrics(text: str) -> list[str]:
    """Detect requested financial metrics."""
    lowered = text.casefold()
    metrics: list[str] = []
    for metric, terms in _METRIC_PATTERNS:
        if any(term in lowered for term in terms):
            metrics.append(metric)
    return metrics


def detect_periods(text: str) -> list[str]:
    """Detect all four-digit financial years."""
    return re.findall(
        r"\b(?:19|20)\d{2}\b",
        text,
    )


def is_narrative_query(text: str) -> bool:
    """Return whether the question asks for an explanation."""
    lowered = text.casefold()
    return any(
        keyword in lowered
        for keyword in _NARRATIVE_KEYWORDS
    )


def analyze_query(
    user_query: str,
    known_companies: list[str] | None = None,
) -> dict[str, Any]:
    """
    Analyze a financial request.
    
    Returns both backward-compatible fields and a normalized task list.
    """
    query = user_query.strip()
    questions = split_questions(query)
    
    tasks: list[dict[str, Any]] = []
    
    for index, question in enumerate(questions, start=1):
        # Detect ALL companies (not just first)
        companies = detect_companies(
            question,
            known_companies=known_companies,
        )
        
        metrics = detect_metrics(question)
        periods = detect_periods(question)
        
        task = {
            "question_id": f"q{index}",
            "question": question,
            "companies": companies,
            "company_name": (
                companies[0] if len(companies) == 1 else None
            ),
            "entities": metrics,
            "metrics": metrics,
            "periods": periods,
            "period": periods[0] if periods else None,
            "is_narrative": is_narrative_query(question),
        }
        tasks.append(task)
    
    all_companies: list[str] = []
    all_metrics: list[str] = []
    all_periods: list[str] = []
    
    for task in tasks:
        for company in task["companies"]:
            if company not in all_companies:
                all_companies.append(company)
        for metric in task["metrics"]:
            if metric not in all_metrics:
                all_metrics.append(metric)
        for period in task["periods"]:
            if period not in all_periods:
                all_periods.append(period)
    
    first_task = tasks[0] if tasks else {}
    
    return {
        "entities": all_metrics,
        "period": all_periods[0] if all_periods else None,
        "periods": all_periods,
        "company_name": (
            all_companies[0] if len(all_companies) == 1 else None
        ),
        "companies": all_companies,
        "is_narrative": any(
            task["is_narrative"]
            for task in tasks
        ),
        "raw_query": user_query,
        "all_metrics": all_metrics,
        "questions": questions,
        "tasks": tasks,
        "question": first_task.get("question", query),
    }


def _ensure_question_mark(text: str) -> str:
    """Normalize question text without changing its meaning."""
    text = text.strip()
    if not text:
        return text
    if text.endswith("?"):
        return text
    return f"{text}?"