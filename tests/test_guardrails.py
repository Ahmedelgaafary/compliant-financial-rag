"""
Tests for GenerationGuard and GuardrailRunner.
"""
# tests/test_guardrails.py
from unittest.mock import Mock

import pytest

from src.guardrails.generation_guard import GenerationGuard
from src.guardrails.policies import GuardrailPolicies
from src.guardrails.runner import GuardrailRunner
from src.retrieval.models import RetrievalResult
from src.verification.models import VerificationResult, VerificationStatus


@pytest.fixture
def policies():
    return GuardrailPolicies(
        min_overall_confidence=0.7,
        min_retrieval_confidence=0.5,
        block_on_numeric_mismatch=True,
        allow_unsupported_claims=False,
    )


@pytest.fixture
def verified_results():
    # Create mock verification results
    #
    # `.status` must be the actual VerificationStatus enum member, not
    # the raw string "VERIFIED" - VerificationStatus is a StrEnum whose
    # value is lowercase ("verified"), and GenerationGuard compares
    # against the enum member directly, so a bare uppercase string
    # never matches and the guard silently treats these as unverified.
    results = []
    v1 = Mock(spec=VerificationResult)
    v1.status = VerificationStatus.VERIFIED
    v1.claim_type = "NUMERIC"
    v1.normalized_value = 42.8
    v1.unit = "billion"
    v1.evidence_chunk_id = "chunk1"
    v1.page_number = 42
    results.append(v1)

    v2 = Mock(spec=VerificationResult)
    v2.status = VerificationStatus.VERIFIED
    v2.claim_type = "TEXT"
    v2.evidence_chunk_id = "chunk2"
    v2.page_number = 10
    results.append(v2)
    return results


def test_generation_guard_blocks_unverified_numbers(policies, verified_results):
    guard = GenerationGuard(policies)
    
    raw_output = "The company reported revenue of $45.2 billion in 2025."
    
    result = guard.guard(raw_output, verified_results)
    
    assert result.is_safe is False
    assert "UNVERIFIED_NUMERIC_CLAIM" in result.issues[0]
    assert "$45.2 billion" in result.flagged_claims


def test_generation_guard_allows_verified_numbers(policies, verified_results):
    guard = GenerationGuard(policies)
    
    raw_output = "The company reported revenue of $42.8 billion."
    
    result = guard.guard(raw_output, verified_results)
    
    assert result.is_safe is True
    assert len(result.issues) == 0


def test_generation_guard_detects_hallucinated_citations(policies, verified_results):
    guard = GenerationGuard(policies)
    
    # Use parentheses to match the citation pattern
    raw_output = "As shown on (Page 99), the revenue increased."
    
    result = guard.guard(raw_output, verified_results)
    
    assert result.is_safe is False
    # Check that "HALLUCINATED_CITATION" appears in any issue
    assert any("HALLUCINATED_CITATION" in issue for issue in result.issues)


def test_guardrail_runner_invalid_input(policies):
    runner = GuardrailRunner(policies)
    
    result = runner.run_full_pipeline(
        query="",  # Empty query
        retrieval_results=[],
        verification_results=[],
        raw_llm_output=None,
    )
    
    assert result.input_valid is False
    assert "INVALID_INPUT_QUERY" in result.output_issues


def test_guardrail_runner_retrieval_fails(policies):
    runner = GuardrailRunner(policies)

    # Create retrieval results with missing provenance
    r1 = Mock(spec=RetrievalResult)
    r1.document_id = None
    r1.chunk_id = "chunk1"
    r1.document_sha256 = None
    r1.score = 0.1
    r1.retrieval_method = "bm25"  # Add missing attribute
    r1.text = "Sample text"
    r1.page_number = 1
    r1.section = "Test"
    
    # Add missing attributes for all mock objects
    r2 = Mock(spec=RetrievalResult)
    r2.document_id = None
    r2.chunk_id = "chunk2"
    r2.document_sha256 = None
    r2.score = 0.1
    r2.retrieval_method = "bm25"
    r2.text = "Sample text"
    r2.page_number = 1
    r2.section = "Test"

    result = runner.run_full_pipeline(
        query="What is revenue?",
        retrieval_results=[r1, r2],
        verification_results=[],
        raw_llm_output=None,
    )

    # Should fail due to missing provenance
    assert result.retrieval_valid is False