"""
Purpose: Provide input, retrieval, and output validators. 
These are the actual guardrails that run at different stages of the workflow.
"""
# src/guardrails/validation.py
import re
from typing import List

from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult

from .policies import GuardrailPolicies


class InputValidator:
    """
    Checks the user query before any processing.
    """
    def __init__(self, policies: GuardrailPolicies):
        self.policies = policies

    def validate(self, query: str) -> bool:
        """Return True if query is valid, False otherwise."""
        # Prevent empty queries
        if not query or not query.strip():
            return False

        # Check for forbidden entities
        for entity in self.policies.forbidden_entities:
            if entity.lower() in query.lower():
                return False

        # (Optional) Detect malformed or malicious patterns
        # For example, SQL injection attempts, excessive length, etc.
        if len(query) > 10000:
            return False

        return True


class RetrievalValidator:
    """
    Checks retrieved evidence before it is passed to the LLM.
    """
    def __init__(self, policies: GuardrailPolicies):
        self.policies = policies

    def validate(
        self,
        retrieval_results: List[RetrievalResult],
    ) -> tuple[bool, List[str]]:
        """
        Returns (is_valid, reasons) for the retrieval result.
        """
        issues = []
        if not retrieval_results:
            issues.append("NO_EVIDENCE_RETRIEVED")
            return False, issues

        # Check that each result has provenance
        for r in retrieval_results:
            if not r.document_id or not r.chunk_id or not r.document_sha256:
                issues.append(f"MISSING_PROVENANCE for chunk {r.chunk_id}")
                return False, issues

        # Check that evidence is sufficient (at least some minimum)
        if len(retrieval_results) < 2:  # arbitrary threshold
            issues.append("INSUFFICIENT_EVIDENCE")

        # Check average retrieval score
        avg_score = sum(r.score for r in retrieval_results) / len(retrieval_results)
        if avg_score < self.policies.min_retrieval_confidence:
            issues.append("LOW_RETRIEVAL_CONFIDENCE")

        return len(issues) == 0, issues


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
    ) -> tuple[bool, List[str]]:
        """
        Ensure the answer does not contain unsupported statements,
        hallucinated citations, or unverified claims.
        """
        issues = []

        # 1. Check that all numeric claims in the answer are supported by verification
        # (This is heuristic: we find numbers and verify they appear in verified claims)
        re.findall(r'\$\d+(?:\.\d+)?[BMK]?', generated_answer)
        # For simplicity, we assume we have a list of verified numeric claims
        # (in real implementation, we'd need to compare values)
        verified_numeric_claims = set()
        for v in verification_results:
            if v.status == "VERIFIED" and "numeric" in v.claim_type.lower():  # pseudo
                verified_numeric_claims.add(v.normalized_value)  # pseudo

        # If the answer has numbers not in verified claims, flag
        # (Simplified – actual implementation would use the claim extraction)
        # Example check:
        # for number in numbers_in_answer:
        #     if number not in verified_numeric_claims:
        #         issues.append("UNVERIFIED_NUMERIC_CLAIM")
        #         break

        # 2. Check for unsupported confidence
        if confidence_score < self.policies.min_overall_confidence:
            issues.append("LOW_CONFIDENCE_ANSWER")

        # 3. Check for citations that might be hallucinated
        # Look for patterns like "(Page X)" or "[citation]" and verify they
        # correspond to actual retrieved chunks.
        # ...

        # 4. If policy says block on numeric mismatch and we found one
        if self.policies.block_on_numeric_mismatch:
            for v in verification_results:
                if v.reason == "NUMERIC_MISMATCH":
                    issues.append("NUMERIC_MISMATCH_BLOCKED")
                    break

        return len(issues) == 0, issues