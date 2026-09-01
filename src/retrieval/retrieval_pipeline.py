"""
Multi-stage retrieval pipeline for financial documents.
Combines hybrid retrieval with exact value search and company fallback.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import RetrievalResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalPipelineConfig:
    """Configuration for the retrieval pipeline."""
    
    # Hybrid retrieval settings
    hybrid_top_k: int = 20
    hybrid_scoped_top_k: int = 10
    
    # Exact value search settings
    exact_value_boost: float = 2.0
    exact_value_threshold: int = 5  # Max results from exact search
    
    # Fallback settings
    enable_fallback: bool = True
    fallback_top_k: int = 5
    fallback_min_chunks: int = 3


class RetrievalPipeline:
    """
    Multi-stage retrieval pipeline.
    
    Stages:
    1. Hybrid retrieval with company scoping
    2. Exact value search for financial numbers
    3. Company fallback search
    """
    
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        chunks: List,
        config: Optional[RetrievalPipelineConfig] = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.chunks = chunks
        self.config = config or RetrievalPipelineConfig()
        
        # Pre-build search indices for exact search
        self._build_exact_search_index()
    
    def _build_exact_search_index(self):
        """Build index for exact value search."""
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}
        self.chunks_by_document = {}
        
        for chunk in self.chunks:
            doc_id = chunk.document_id
            if doc_id not in self.chunks_by_document:
                self.chunks_by_document[doc_id] = []
            self.chunks_by_document[doc_id].append(chunk)
    
    def retrieve(
        self,
        query: str,
        company: Optional[str] = None,
        metric: Optional[str] = None,
        period: Optional[str] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most relevant chunks using multi-stage pipeline.
        """
        logger.info(f"Retrieval pipeline: query='{query[:50]}...', company={company}, metric={metric}, period={period}")
        
        all_results = []
        seen_chunk_ids = set()
        
        # ============================================================
        # STAGE 1: Hybrid Retrieval
        # ============================================================
        hybrid_results = self._stage_hybrid_retrieval(query, company, top_k=20)
        all_results.extend(hybrid_results)
        seen_chunk_ids.update(r.chunk_id for r in hybrid_results)
        
        # ============================================================
        # STAGE 2: Exact Value Search
        # ============================================================
        if metric or period:
            exact_results = self._stage_exact_value_search(
                query, company, metric, period
            )
            # Add exact results that aren't already in the list
            for r in exact_results:
                if r.chunk_id not in seen_chunk_ids:
                    all_results.append(r)
                    seen_chunk_ids.add(r.chunk_id)
        
        # ============================================================
        # STAGE 3: Company Fallback
        # ============================================================
        if company and len(all_results) < self.config.hybrid_scoped_top_k:
            fallback_results = self._stage_company_fallback(company, period)
            for r in fallback_results:
                if r.chunk_id not in seen_chunk_ids:
                    all_results.append(r)
                    seen_chunk_ids.add(r.chunk_id)
        
        # Remove duplicates and sort by score
        unique_results = self._deduplicate_and_sort(all_results)
        
        # Return top_k
        return unique_results[:top_k]
    
    def _stage_hybrid_retrieval(
        self,
        query: str,
        company: Optional[str],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Stage 1: Hybrid retrieval."""
        results = list(self.hybrid_retriever.retrieve(query, top_k=top_k))
        
        # If company is specified, scope results
        if company:
            from src.agent.node import _result_matches_company
            scoped = [r for r in results if _result_matches_company(r, company)]
            if scoped:
                results = scoped
        
        logger.info(f"Stage 1 (Hybrid): {len(results)} results")
        return results
    
    def _stage_exact_value_search(
        self,
        query: str,
        company: Optional[str],
        metric: Optional[str],
        period: Optional[str],
    ) -> List[RetrievalResult]:
        """
        Stage 2: Exact value search.
        
        Searches for exact financial values in the corpus.
        This catches chunks that hybrid retrieval might miss.
        """
        results = []
        
        # Extract potential values from query
        values_to_search = self._extract_values_from_query(query)
        
        # Common financial patterns
        patterns = [
            r"Total\s+net\s+sales\s+\$\s*([\d,]+)",
            r"Total\s+revenue\s+\$\s*([\d,]+)",
            r"Net\s+sales\s+\$\s*([\d,]+)",
            r"Revenue\s+\$\s*([\d,]+)",
        ]
        
        # Search all chunks
        for chunk in self.chunks:
            text = chunk.text
            
            # Skip if not the right company
            if company and company.lower() not in chunk.document_id.lower():
                continue
            
            # Skip if not the right period
            if period and period not in text:
                # Check if period is in the text
                if period not in text:
                    continue
            
            # Check for value patterns
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # This chunk contains a financial value
                    score = 1.0
                    result = RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        score=score,
                        page_number=chunk.page_number,
                        section=chunk.section,
                        document_sha256=chunk.document_sha256,
                        retrieval_method="exact_value_search",
                    )
                    results.append(result)
                    break
        
        # Check if any of the exact values from query appear
        for value in values_to_search:
            for chunk in self.chunks:
                if value in chunk.text:
                    if company and company.lower() not in chunk.document_id.lower():
                        continue
                    result = RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        score=0.9,
                        page_number=chunk.page_number,
                        section=chunk.section,
                        document_sha256=chunk.document_sha256,
                        retrieval_method="exact_value_search",
                    )
                    if result.chunk_id not in [r.chunk_id for r in results]:
                        results.append(result)
        
        logger.info(f"Stage 2 (Exact Value): {len(results)} results")
        return results[:self.config.exact_value_threshold]
    
    def _stage_company_fallback(
        self,
        company: str,
        period: Optional[str],
    ) -> List[RetrievalResult]:
        """
        Stage 3: Company fallback search.
        
        If not enough results, search specifically for the company's document.
        """
        results = []
        
        # Find documents for this company
        for doc_id, chunks in self.chunks_by_document.items():
            if company.lower() in doc_id.lower():
                # If period is specified, prefer that year
                if period:
                    if period not in doc_id:
                        continue
                
                # Take the first few chunks from this document
                for chunk in chunks[:self.config.fallback_top_k]:
                    result = RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        score=0.7,
                        page_number=chunk.page_number,
                        section=chunk.section,
                        document_sha256=chunk.document_sha256,
                        retrieval_method="company_fallback",
                    )
                    results.append(result)
                break
        
        logger.info(f"Stage 3 (Fallback): {len(results)} results")
        return results
    
    def _extract_values_from_query(self, query: str) -> List[str]:
        """Extract potential financial values from the query."""
        values = []
        
        # Check for specific company revenue values
        company_value_patterns = [
            r"Apple\s+revenue\s+(\d+)",
            r"Microsoft\s+revenue\s+(\d+)",
            r"revenue\s+(\d+\.?\d*)\s*(?:million|billion|B|M)",
        ]
        
        for pattern in company_value_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                values.append(match.group(1))
        
        return values
    
    def _deduplicate_and_sort(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Remove duplicates and sort by score."""
        seen = set()
        unique = []
        
        # Sort by score (highest first)
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        
        for r in sorted_results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                unique.append(r)
        
        return unique