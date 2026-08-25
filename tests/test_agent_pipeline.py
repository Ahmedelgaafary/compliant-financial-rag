"""
Consolidated test suite for the agent pipeline.
Run with: pytest tests/test_agent_pipeline.py -v
"""

from datetime import datetime
from unittest.mock import patch

import pytest
from src.agent.graph import build_agent_graph
from src.agent.node import (
    answer_generation_node,
    audit_node,
    claim_generation_node,
    guardrail_node,
    query_analysis_node,
    retrieval_node,
    routing_node,
    verification_node,
)

from src.agent.state import AgentState
from src.audit.models import AuditRecord
from src.audit.review_service import ReviewOutcome
from src.audit.router import RoutingAction, RoutingDecision
from src.guardrails.risk_engine import RiskAssessment
from src.guardrails.runner import GuardrailPipelineResult
from src.retrieval.models import RetrievalResult
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def base_state():
    return AgentState(user_query="What was revenue in 2025?")


@pytest.fixture
def mock_retrieval_results():
    return [
        RetrievalResult(
            chunk_id="chunk1",
            document_id="doc1",
            text="Revenue for 2025 was $42.8 billion.",
            score=0.9,
            page_number=3,
            section="Financials",
            document_sha256="abc123",
            retrieval_method="hybrid_rrf",
        )
    ]


@pytest.fixture
def mock_claim():
    return Claim(
        claim_id="claim1",
        claim_type=ClaimType.NUMERIC,
        subject="revenue",
        value="$42.8B",
        unit="billion",
        period="2025",
        source_chunk_id="chunk1",
    )


@pytest.fixture
def mock_verification_result(mock_claim):
    return VerificationResult(
        claim_id=mock_claim.claim_id,
        status=VerificationStatus.VERIFIED,
        reason="numeric_match",
        confidence=0.95,
        evidence_chunk_id="chunk1",
    )


# Helper functions for constructing objects
def build_guardrail_result(
    risk_assessment: RiskAssessment,
    should_route_to_audit: bool
) -> GuardrailPipelineResult:
    return GuardrailPipelineResult(
        input_valid=True,
        retrieval_valid=True,
        retrieval_issues=[],
        confidence_score=None,
        risk_assessment=risk_assessment,
        should_route_to_audit=should_route_to_audit,
        generation_guard_result=None,
        output_valid=True,
        output_issues=[],
        final_safe_output="Generated answer",
    )


def build_routing_decision(
    should_create_audit_record: bool,
    action: RoutingAction,
    reason: str = "Risk threshold exceeded",
    audit_priority: int = 1,
) -> RoutingDecision:
    return RoutingDecision(
        should_create_audit_record=should_create_audit_record,
        action=action,
        reason=reason,
        audit_priority=audit_priority,
    )


# ----------------------------------------------------------------------
# Unit Tests – Each Node
# ----------------------------------------------------------------------

def test_query_analysis_node_sets_analysis(base_state):
    result = query_analysis_node(base_state)
    assert result.query_analysis is not None
    assert "entities" in result.query_analysis


def test_retrieval_node_uses_retriever(base_state, mock_retrieval_results):
    with patch("src.agent.node.HybridRetriever") as mock_retriever_cls:
        mock_retriever = mock_retriever_cls.return_value
        mock_retriever.retrieve.return_value = mock_retrieval_results
        result = retrieval_node(base_state)
        assert result.retrieval_results == mock_retrieval_results


def test_claim_generation_node_extracts_claim(base_state, mock_retrieval_results):
    base_state.retrieval_results = mock_retrieval_results
    with patch("src.agent.node.LLMClient") as mock_llm_cls:
        mock_llm = mock_llm_cls.return_value
        mock_llm.generate.return_value = "revenue = $42.8B"
        result = claim_generation_node(base_state)
        assert result.raw_llm_output == "revenue = $42.8B"
        assert len(result.claims) == 1
        assert result.claims[0].subject == "revenue"
        assert result.claims[0].unit == "billion"  # abbreviation mapped


def test_verification_node_calls_verifier(
    base_state, mock_claim, mock_retrieval_results, mock_verification_result
):
    base_state.claims = [mock_claim]
    base_state.retrieval_results = mock_retrieval_results
    with patch("src.agent.node.Verifier") as mock_verifier_cls:
        mock_verifier = mock_verifier_cls.return_value
        mock_verifier.verify.return_value = mock_verification_result
        result = verification_node(base_state)
        assert result.verification_results == [mock_verification_result]
        mock_verifier.verify.assert_called_once_with(
            mock_claim, "Revenue for 2025 was $42.8 billion."
        )


def test_guardrail_node_sets_risk_and_audit_flag(
    base_state, mock_retrieval_results, mock_verification_result
):
    base_state.retrieval_results = mock_retrieval_results
    base_state.verification_results = [mock_verification_result]
    base_state.raw_llm_output = "some claim"
    with patch("src.agent.node.GuardrailRunner") as mock_runner_cls:
        mock_runner = mock_runner_cls.return_value
        risk = RiskAssessment(
            risk_score=0.8,
            risk_level="HIGH",
            triggers=["NUMERIC_MISMATCH"],
            recommended_action="HUMAN_REVIEW",
        )
        mock_runner.run_full_pipeline.return_value = build_guardrail_result(
            risk_assessment=risk,
            should_route_to_audit=True,
        )
        result = guardrail_node(base_state)
        assert result.risk_assessment.risk_level == "HIGH"
        assert result.should_route_to_audit is True


def test_routing_node_returns_state(base_state):
    result = routing_node(base_state)
    assert result is base_state


def test_answer_generation_uses_verified_claims(
    base_state, mock_retrieval_results, mock_verification_result
):
    base_state.retrieval_results = mock_retrieval_results
    base_state.verification_results = [mock_verification_result]
    with patch("src.agent.node.LLMClient") as mock_llm_cls:
        mock_llm = mock_llm_cls.return_value
        mock_llm.generate.return_value = "The revenue was $42.8B in 2025."
        result = answer_generation_node(base_state)
        assert "verified claims" in mock_llm.generate.call_args[0][0].lower()
        assert result.final_answer == "The revenue was $42.8B in 2025."


def test_audit_node_calls_review_service(
    base_state, mock_retrieval_results, mock_verification_result
):
    base_state.retrieval_results = mock_retrieval_results
    base_state.verification_results = [mock_verification_result]
    base_state.risk_assessment = RiskAssessment(
        risk_score=0.8,
        risk_level="HIGH",
        triggers=[],
        recommended_action="HUMAN_REVIEW",
    )
    with patch("src.agent.node.ReviewService") as mock_review_cls:
        mock_review = mock_review_cls.return_value
        audit_record = AuditRecord(
            audit_id="AUD-1",
            timestamp=datetime.now(),
            created_at=datetime.now(),
            user_query="What was revenue in 2025?",
            claim="revenue = $42.8B",
            verification_status="verified",
            verification_reason="numeric_match",
            risk_level="HIGH",
            risk_assessment=base_state.risk_assessment,
            evidence=[],
            document_id="doc1",
            document_sha256="abc123",
            page_number=3,
            confidence_score=0.8,
            risk_score=0.8,
            triggers=[],
            verification_results=[],
        )
        mock_review.initiate_review.return_value = ReviewOutcome(
            audit_id="AUD-1",
            routing_decision=build_routing_decision(
                should_create_audit_record=True,
                action=RoutingAction.HUMAN_REVIEW,
                reason="Risk threshold exceeded",
                audit_priority=2,
            ),
            review_recommendation=None,
            final_action="HUMAN_REVIEW",
            audit_record=audit_record,
        )
        result = audit_node(base_state)
        assert result.audit_record == audit_record
        assert "human review" in result.final_answer.lower()


# ----------------------------------------------------------------------
# Integration Tests – Full Graph
# ----------------------------------------------------------------------

def test_graph_low_risk_returns_answer():
    with patch("src.agent.node.HybridRetriever") as mock_retriever, \
         patch("src.agent.node.LLMClient") as mock_llm, \
         patch("src.agent.node.Verifier") as mock_verifier, \
         patch("src.agent.node.GuardrailRunner") as mock_guardrail, \
         patch("src.agent.node.ReviewService") as mock_review:
        # Setup mocks
        mock_retriever.return_value.retrieve.return_value = [
            RetrievalResult(
                chunk_id="chunk1",
                document_id="doc1",
                text="Revenue for 2025 was $42.8 billion.",
                score=0.9,
                page_number=3,
                section="Financials",
                document_sha256="abc123",
                retrieval_method="hybrid_rrf",
            )
        ]
        mock_llm.return_value.generate.side_effect = [
            "revenue = $42.8B",  # claim generation
            "The revenue was $42.8B in 2025."  # answer generation
        ]
        mock_verifier.return_value.verify.return_value = VerificationResult(
            claim_id="claim1",
            status=VerificationStatus.VERIFIED,
            reason="numeric_match",
            confidence=0.95,
            evidence_chunk_id="chunk1",
        )
        risk_low = RiskAssessment(
            risk_score=0.1,
            risk_level="LOW",
            triggers=[],
            recommended_action="AUTO_ANSWER",
        )
        mock_guardrail.return_value.run_full_pipeline.return_value = (
            build_guardrail_result(
                risk_assessment=risk_low,
                should_route_to_audit=False,
            )
        )
        graph = build_agent_graph()
        result = graph.invoke({"user_query": "What was revenue in 2025?"})
        assert result["final_answer"] == "The revenue was $42.8B in 2025."
        mock_review.return_value.initiate_review.assert_not_called()


def test_graph_high_risk_routes_to_audit():
    with patch("src.agent.node.HybridRetriever") as mock_retriever, \
         patch("src.agent.node.LLMClient") as mock_llm, \
         patch("src.agent.node.Verifier") as mock_verifier, \
         patch("src.agent.node.GuardrailRunner") as mock_guardrail, \
         patch("src.agent.node.ReviewService") as mock_review:
        # High‑risk setup
        mock_retriever.return_value.retrieve.return_value = []
        mock_llm.return_value.generate.side_effect = ["", ""]  # no claim
        mock_verifier.return_value.verify.return_value = None  # will not be called
        risk_high = RiskAssessment(
            risk_score=0.9,
            risk_level="HIGH",
            triggers=["INSUFFICIENT_EVIDENCE"],
            recommended_action="HUMAN_REVIEW",
        )
        mock_guardrail.return_value.run_full_pipeline.return_value = (
            build_guardrail_result(
                risk_assessment=risk_high,
                should_route_to_audit=True,
            )
        )
        audit_record = AuditRecord(
            audit_id="AUD-1",
            timestamp=datetime.now(),
            created_at=datetime.now(),
            user_query="What was revenue in 2025?",
            claim="No claim extracted",
            verification_status="inconclusive",
            verification_reason="EVIDENCE_MISSING",
            risk_level="HIGH",
            risk_assessment=risk_high,
            evidence=[],
            document_id="",
            document_sha256="",
            page_number=0,
            confidence_score=0.9,
            risk_score=0.9,
            triggers=["INSUFFICIENT_EVIDENCE"],
            verification_results=[],
        )
        mock_review.return_value.initiate_review.return_value = ReviewOutcome(
            audit_id="AUD-1",
            routing_decision=build_routing_decision(
                should_create_audit_record=True,
                action=RoutingAction.HUMAN_REVIEW,
                reason="Insufficient evidence",
                audit_priority=1,
            ),
            review_recommendation=None,
            final_action="HUMAN_REVIEW",
            audit_record=audit_record,
        )
        graph = build_agent_graph()
        result = graph.invoke({"user_query": "What was revenue in 2025?"})
        assert "human review" in result["final_answer"].lower()
        mock_review.return_value.initiate_review.assert_called_once()