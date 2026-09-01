"""
Reranker module for improving retrieval quality.
Reranks retrieved chunks based on relevance to the query and company.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from src.retrieval.models import RetrievalResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Result of reranking."""
    chunk_id: str
    document_id: str
    text: str
    original_score: float
    rerank_score: float
    page_number: int
    section: str
    document_sha256: str
    retrieval_method: str


class FinancialReranker:
    """
    Reranks financial document chunks based on relevance signals.
    
    Features:
    - Exact financial value matching (e.g., "416,161")
    - Metric keyword matching (e.g., "revenue", "net sales")
    - Period matching (e.g., "2025")
    - Company name matching
    - Financial table detection
    - Page proximity bonus (financial statements are usually pages 25-45)
    """
    
    # Financial metrics and their variations
    METRIC_KEYWORDS = {
        "revenue": ["revenue", "net sales", "total net sales", "sales", "turnover"],
        "income": ["net income", "income", "earnings", "net earnings", "profit"],
        "margin": ["margin", "gross margin", "operating margin", "net margin"],
        "assets": ["assets", "total assets", "current assets"],
        "liabilities": ["liabilities", "total liabilities", "current liabilities"],
        "cash": ["cash flow", "operating cash flow", "cash"],
        "ebitda": ["ebitda", "EBITDA"],
    }
    
    # Financial statement pages are typically in this range
    FINANCIAL_STATEMENT_PAGES = range(25, 50)
    
    def __init__(
        self,
        boost_exact_value: float = 0.5,
        boost_metric: float = 0.3,
        boost_period: float = 0.2,
        boost_company: float = 0.2,
        boost_financial_page: float = 0.4,
        boost_table: float = 0.3,
    ):
        """
        Initialize the reranker with boost weights.
        
        Args:
            boost_exact_value: Boost for chunks containing exact financial values
            boost_metric: Boost for chunks containing metric keywords
            boost_period: Boost for chunks containing the requested period
            boost_company: Boost for chunks containing the company name
            boost_financial_page: Boost for chunks on financial statement pages
            boost_table: Boost for chunks containing financial tables
        """
        self.boost_exact_value = boost_exact_value
        self.boost_metric = boost_metric
        self.boost_period = boost_period
        self.boost_company = boost_company
        self.boost_financial_page = boost_financial_page
        self.boost_table = boost_table
    
    def rerank(
        self,
        results: List[RetrievalResult],
        query: str,
        company: Optional[str] = None,
        metric: Optional[str] = None,
        period: Optional[str] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results based on multiple relevance signals.
        
        Args:
            results: List of retrieval results to rerank
            query: The original user query
            company: The company name (e.g., "Apple")
            metric: The financial metric (e.g., "revenue")
            period: The fiscal period (e.g., "2025")
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked list of RetrievalResult objects
        """
        if not results:
            return []
        
        logger.info(f"Reranking {len(results)} results for query: {query[:50]}...")
        
        # Parse query for signals
        query_lower = query.lower()
        if not metric:
            metric = self._detect_metric(query_lower)
        if not period:
            period = self._detect_period(query_lower)
        if not company:
            company = self._detect_company(query_lower)
        
        # Score each result
        scored_results = []
        for result in results:
            rerank_score = self._compute_rerank_score(
                result=result,
                query=query_lower,
                company=company,
                metric=metric,
                period=period,
            )
            
            scored_results.append(
                RerankResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    text=result.text,
                    original_score=result.score,
                    rerank_score=rerank_score,
                    page_number=result.page_number,
                    section=result.section,
                    document_sha256=result.document_sha256,
                    retrieval_method=result.retrieval_method,
                )
            )
        
        # Sort by rerank score (descending)
        scored_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Convert back to RetrievalResult
        reranked = []
        for sr in scored_results[:top_k]:
            reranked.append(
                RetrievalResult(
                    chunk_id=sr.chunk_id,
                    document_id=sr.document_id,
                    text=sr.text,
                    score=sr.rerank_score,  # Use rerank score as new score
                    page_number=sr.page_number,
                    section=sr.section,
                    document_sha256=sr.document_sha256,
                    retrieval_method=f"reranked_{sr.retrieval_method}",
                )
            )
        
        logger.info(f"Reranking complete: {len(reranked)} results returned")
        return reranked
    
    def _compute_rerank_score(
        self,
        result: RetrievalResult,
        query: str,
        company: Optional[str],
        metric: Optional[str],
        period: Optional[str],
    ) -> float:
        """
        Compute a combined rerank score for a single result.
        """
        text = result.text.lower()
        doc_id = result.document_id.lower() if result.document_id else ""
        
        # Start with original score (normalized)
        score = min(result.score, 1.0)
        
        # 1. Exact value bonus (e.g., "416,161" in the text)
        exact_value_bonus = self._compute_exact_value_bonus(text)
        score += exact_value_bonus * self.boost_exact_value
        
        # 2. Metric bonus
        metric_bonus = self._compute_metric_bonus(text, metric)
        score += metric_bonus * self.boost_metric
        
        # 3. Period bonus
        period_bonus = self._compute_period_bonus(text, period)
        score += period_bonus * self.boost_period
        
        # 4. Company bonus
        company_bonus = self._compute_company_bonus(text, doc_id, company)
        score += company_bonus * self.boost_company
        
        # 5. Financial page bonus
        page_bonus = self._compute_page_bonus(result.page_number)
        score += page_bonus * self.boost_financial_page
        
        # 6. Table detection bonus
        table_bonus = self._compute_table_bonus(text)
        score += table_bonus * self.boost_table
        
        # Do not cap the upper bound: this score exists purely to
        # support relative sorting within rerank(), not to act as a
        # bounded probability. With six binary bonus signals summing
        # to a max weight of 1.9 against a tiny base retrieval score
        # (real hybrid/BM25 scores are often ~0.01-0.03), clamping to
        # 1.0 meant any chunk matching just 3-4 signals saturated to
        # the same ceiling as a chunk matching all of them - making
        # the reranker unable to distinguish a well-matched chunk from
        # a mediocre one, which is how a wrong-year table could
        # outrank (or tie with, and win by arbitrary original order)
        # the actually correct evidence. Downstream confidence scoring
        # already clamps scores safely on its own, so an unbounded
        # value here is not a problem for callers.
        return max(0.0, score)
    
    def _compute_exact_value_bonus(self, text: str) -> float:
        """
        Check for exact financial values in the text.
        """
        # Look for large numbers with $ signs
        patterns = [
            r'\$\s*[\d,]+\.?\d*\s*(?:million|billion|B|M)',  # $416,161 million
            r'\$\s*[\d,]+\.?\d*',  # $416,161
            r'total net sales\s+\$\s*[\d,]+',  # Total net sales $ 416,161
            r'revenue\s+\$\s*[\d,]+',  # Revenue $ 416,161
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
        return 0.0
    
    def _compute_metric_bonus(self, text: str, metric: Optional[str]) -> float:
        """
        Check for metric keywords in the text.
        """
        if not metric:
            return 0.0
        
        # Get keyword variations for this metric
        keywords = self.METRIC_KEYWORDS.get(metric.lower(), [metric.lower()])
        
        # Check if any keyword is in the text
        for keyword in keywords:
            if keyword in text:
                return 1.0
        
        return 0.0
    
    def _compute_period_bonus(self, text: str, period: Optional[str]) -> float:
        """
        Check for the requested period in the text.

        A bare substring check on the whole chunk rewards incidental
        mentions of the period anywhere in the text (a footer, an
        unrelated sentence) just as much as a genuine table header for
        that fiscal year. Prefer a period that appears near a currency
        figure or "(in millions)"-style table annotation - that's a
        much stronger signal this chunk's numbers actually belong to
        the requested year, rather than just mentioning it in passing.
        """
        if not period:
            return 0.0

        for match in re.finditer(re.escape(period), text):
            window = text[
                max(0, match.start() - 100) : match.end() + 100
            ]

            if re.search(r"\$\s*[\d,]", window) or re.search(
                r"\(\s*in\s+(?:thousands?|millions?|billions?|trillions?)\s*\)",
                window,
                re.IGNORECASE,
            ):
                return 1.0

        # Fall back to a weaker signal for a bare mention anywhere -
        # still worth something, just less than a table-adjacent one.
        if period in text:
            return 0.4

        if re.search(rf"fiscal year.*{period}", text, re.IGNORECASE):
            return 0.3

        return 0.0

    def _compute_company_bonus(
        self,
        text: str,
        doc_id: str,
        company: Optional[str],
    ) -> float:
        """
        Check for company name in text or document ID.
        """
        if not company:
            return 0.0
        
        company_lower = company.lower()
        
        # Check in document ID
        if company_lower in doc_id:
            return 1.0
        
        # Check in text
        if company_lower in text:
            return 0.8
        
        # Check for company variations
        from src.config.companies import CompanyConfig
        config = CompanyConfig.get_company(company)
        if config:
            for variation in config.get("variations", []):
                if variation.lower() in text:
                    return 0.6
        
        return 0.0
    
    def _compute_page_bonus(self, page_number: int) -> float:
        """
        Check if the page is in the financial statement range.
        """
        if page_number in self.FINANCIAL_STATEMENT_PAGES:
            return 1.0
        elif 20 <= page_number <= 60:
            return 0.5
        return 0.0
    
    def _compute_table_bonus(self, text: str) -> float:
        """
        Detect if the text contains a financial table.

        "Total net sales" (and similar) previously matched as a bare
        substring, so a percentage row like "Percentage of total net
        sales 14% 13% 12%" scored the same as an actual dollar-value
        table row. Require the phrase to be followed by a currency
        amount so this bonus reflects a genuine data row rather than
        any sentence that happens to contain the words.
        """
        table_indicators = [
            r'\(in millions\)',
            r'\(in billions\)',
            r'\(in thousands\)',
            r'table shows',
            r'disaggregated',
            r'(?:total\s+net\s+sales|total\s+revenue|net\s+income)\s+\$\s*[\d,]',
            r'\[\s*[\d,]+\s*\]',  # Numbers in brackets
            r'\$\s*[\d,]+\s+\$\s*[\d,]+\s+\$\s*[\d,]+',  # Multiple dollar values
        ]
        
        for indicator in table_indicators:
            if re.search(indicator, text, re.IGNORECASE):
                return 1.0
        
        return 0.0
    
    def _detect_metric(self, query: str) -> Optional[str]:
        """
        Detect the financial metric from the query.
        """
        for metric, keywords in self.METRIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    return metric
        return None
    
    def _detect_period(self, query: str) -> Optional[str]:
        """
        Detect the fiscal period from the query.
        """
        match = re.search(r'\b(19|20)\d{2}\b', query)
        return match.group(0) if match else None
    
    def _detect_company(self, query: str) -> Optional[str]:
        """
        Detect the company from the query.
        """
        from src.config.companies import CompanyConfig
        return CompanyConfig.detect_company(query)