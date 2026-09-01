"""
Purpose: Provide input, retrieval, and output validators.

These validators enforce guardrail checks at different stages of the
agent workflow.

Important:
    RetrievalResult.score does not always represent a probability or
    confidence score. In particular, hybrid retrieval uses Reciprocal
    Rank Fusion (RRF), whose score scale is much smaller than [0, 1].

    Therefore, this module converts hybrid RRF scores into a normalized
    confidence signal before comparing them with the configured
    min_retrieval_confidence threshold.
"""

import logging
from typing import List, Tuple

from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult

from .policies import GuardrailPolicies

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Checks the user query before any processing.
    """

    def __init__(self, policies: GuardrailPolicies):
        self.policies = policies

    def validate(self, query: str) -> bool:
        """
        Return True if the query is valid, otherwise False.
        """
        if not query or not query.strip():
            logger.warning("Empty query rejected")
            return False

        for entity in self.policies.forbidden_entities:
            if entity.lower() in query.lower():
                logger.warning(f"Query contains forbidden entity: {entity}")
                return False

        if len(query) > 10000:
            logger.warning(f"Query too long: {len(query)} chars")
            return False

        return True


class RetrievalValidator:
    """
    Checks retrieved evidence before it is passed to the LLM.

    The validator handles different retrieval score semantics with
    automatic normalization:
    
    - hybrid_rrf: Normalized against theoretical maximum RRF score
    - vector: Cosine similarity converted to [0, 1]
    - bm25: Requires evidence presence, not probabilistic scores
    """

    def __init__(self, policies: GuardrailPolicies):
        self.policies = policies
        self._rrf_k = 60  # Standard RRF constant
        self._max_rrf_score = 2.0 / (self._rrf_k + 1)  # ~0.03279

    def validate(
        self,
        retrieval_results: List[RetrievalResult],
    ) -> Tuple[bool, List[str]]:
        """
        Return (is_valid, issues) for the retrieved evidence.
        """
        issues: List[str] = []

        # 1. Evidence must exist.
        if not retrieval_results:
            issues.append("NO_EVIDENCE_RETRIEVED")
            logger.error("No evidence retrieved")
            return False, issues

        # 2. Every result must have complete provenance.
        for result in retrieval_results:
            if not result.document_id:
                issues.append(f"MISSING_DOCUMENT_ID for chunk {result.chunk_id}")
                logger.error(f"Missing document_id: {result.chunk_id}")
                return False, issues
            
            if not result.chunk_id:
                issues.append("MISSING_CHUNK_ID")
                logger.error("Missing chunk_id")
                return False, issues
                
            if not result.document_sha256:
                issues.append(f"MISSING_DOCUMENT_SHA256 for chunk {result.chunk_id}")
                logger.error(f"Missing document_sha256: {result.chunk_id}")
                return False, issues

            if not result.text or not result.text.strip():
                issues.append(f"EMPTY_EVIDENCE_TEXT for chunk {result.chunk_id}")
                logger.error(f"Empty evidence text: {result.chunk_id}")
                return False, issues

        # 3. Check evidence quantity.
        if len(retrieval_results) < self.policies.min_evidence_chunks:
            issues.append("INSUFFICIENT_EVIDENCE")
            logger.warning(f"Only {len(retrieval_results)} chunks, need {self.policies.min_evidence_chunks}")

        # 4. Evaluate retrieval confidence with automatic normalization.
        confidence = self._compute_retrieval_confidence(retrieval_results)
        logger.info(f"Retrieval confidence: {confidence:.3f}, threshold: {self.policies.min_retrieval_confidence}")

        if confidence < self.policies.min_retrieval_confidence:
            issues.append("LOW_RETRIEVAL_CONFIDENCE")
            logger.warning(
                f"Retrieval confidence {confidence:.3f} below threshold "
                f"{self.policies.min_retrieval_confidence}"
            )
        
        is_valid = len(issues) == 0
        if is_valid:
            logger.info(f"Retrieval validation PASSED: {len(retrieval_results)} chunks, confidence {confidence:.3f}")
        else:
            logger.warning(f"Retrieval validation FAILED: issues={issues}")
        
        return is_valid, issues

    def _compute_retrieval_confidence(
        self,
        retrieval_results: List[RetrievalResult],
    ) -> float:
        """
        Convert retrieval scores into a normalized confidence value.
        
        Returns a value in [0, 1] representing retrieval quality.
        """
        if not retrieval_results:
            return 0.0

        methods = {result.retrieval_method for result in retrieval_results}
        logger.debug(f"Retrieval methods detected: {methods}")

        # Hybrid RRF - Normalize against theoretical maximum
        if "hybrid_rrf" in methods:
            normalized_scores = []
            for result in retrieval_results:
                if result.retrieval_method == "hybrid_rrf":
                    # Normalize RRF score against theoretical maximum
                    normalized = result.score / self._max_rrf_score
                    normalized = min(1.0, max(0.0, normalized))
                    normalized_scores.append(normalized)
                    logger.debug(f"RRF score {result.score:.6f} -> normalized {normalized:.3f}")
            
            if normalized_scores:
                confidence = self._weighted_top_k_confidence(normalized_scores)
                logger.info(f"Hybrid RRF confidence: {confidence:.3f}")
                return confidence

        # Vector retrieval - Cosine similarity to [0, 1]
        if methods == {"vector"} or "vector" in methods:
            normalized_scores = [
                self._normalize_cosine_similarity(result.score)
                for result in retrieval_results
            ]
            confidence = self._weighted_top_k_confidence(normalized_scores)
            logger.info(f"Vector confidence: {confidence:.3f}")
            return confidence

        # BM25-only - Scores are corpus-dependent, use presence-based confidence
        if methods == {"bm25"}:
            # High confidence if we have good evidence quantity
            confidence = min(1.0, len(retrieval_results) / self.policies.min_evidence_chunks)
            logger.info(f"BM25 confidence (presence-based): {confidence:.3f}")
            return confidence

        # Unknown/mixed - Conservative fallback
        logger.warning(f"Unknown retrieval methods: {methods}, using conservative confidence")
        normalized_scores = [
            min(1.0, max(0.0, float(result.score)))
            for result in retrieval_results
        ]
        return self._weighted_top_k_confidence(normalized_scores)

    @staticmethod
    def _normalize_cosine_similarity(score: float) -> float:
        """Convert cosine similarity from [-1, 1] to [0, 1]."""
        normalized = (float(score) + 1.0) / 2.0
        return min(1.0, max(0.0, normalized))

    @staticmethod
    def _weighted_top_k_confidence(scores: List[float]) -> float:
        """
        Calculate deterministic confidence emphasizing strongest evidence.
        
        Weights:
            rank 1 -> 0.50
            rank 2 -> 0.30
            remaining -> 0.20 distributed equally
        """
        if not scores:
            return 0.0

        ordered_scores = sorted(scores, reverse=True)

        if len(ordered_scores) == 1:
            return ordered_scores[0]

        if len(ordered_scores) == 2:
            return 0.60 * ordered_scores[0] + 0.40 * ordered_scores[1]

        top_score = ordered_scores[0]
        second_score = ordered_scores[1]
        remaining_scores = ordered_scores[2:]

        remaining_average = sum(remaining_scores) / len(remaining_scores) if remaining_scores else 0.0

        confidence = 0.50 * top_score + 0.30 * second_score + 0.20 * remaining_average
        return min(1.0, max(0.0, confidence))


class OutputValidator:
    """
    Final check on the generated answer before returning it to the user.
    """

    def __init__(self, policies: GuardrailPolicies):
        self.policies = policies

    def validate(
        self,
        generated_answer: str,
        verification_results: List[VerificationResult],
        confidence_score: float,
    ) -> Tuple[bool, List[str]]:
        """
        Ensure the generated answer does not contain unsupported
        statements, hallucinated citations, or unverified claims.
        """
        issues: List[str] = []

        # Generated answer must have sufficient confidence.
        if confidence_score < self.policies.min_overall_confidence:
            issues.append("LOW_CONFIDENCE_ANSWER")
            logger.warning(
                f"Answer confidence {confidence_score:.3f} below threshold "
                f"{self.policies.min_overall_confidence}"
            )
        # Numeric mismatch blocking.
        if self.policies.block_on_numeric_mismatch:
            for result in verification_results:
                reason = str(getattr(result, "reason", "")).upper()
                if reason == "NUMERIC_MISMATCH":
                    issues.append("NUMERIC_MISMATCH_BLOCKED")
                    logger.error("Numeric mismatch detected, blocking answer")
                    break

        is_valid = len(issues) == 0
        logger.info(f"Output validation: {'PASSED' if is_valid else 'FAILED'}")
        return is_valid, issues