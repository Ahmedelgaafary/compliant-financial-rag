"""
Consolidated test suite for the agent pipeline.

Run:
    pytest tests/test_agent_pipeline.py -v

This suite covers:
    1. Query analysis
    2. Hybrid retrieval
    3. Deterministic claim generation
    4. Verification
    5. Guardrails
    6. Routing
    7. Answer generation
    8. Audit handling
    9. Full graph execution
    10. Output safety blocking
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
from src.ingestion.chunker import DocumentChunk
from src.retrieval.models import RetrievalResult
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationResult,
    VerificationStatus,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_state() -> AgentState:
    """Create the minimum valid agent state."""
    return AgentState(
        user_query="What was revenue in 2025?",
    )


@pytest.fixture
def mock_retrieval_results() -> list[RetrievalResult]:
    """Create deterministic financial evidence."""
    return [
        RetrievalResult(
            chunk_id="chunk1",
            document_id="doc1",
            text="Revenue was $42.8 billion for fiscal year 2025.",
            score=0.9,
            page_number=3,
            section="Financials",
            document_sha256="abc123",
            retrieval_method="hybrid_rrf",
        )
    ]


@pytest.fixture
def mock_claim() -> Claim:
    """Create a verified numeric revenue claim."""
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
def mock_verification_result(
    mock_claim: Claim,
) -> VerificationResult:
    """Create a successful verification result."""
    return VerificationResult(
        claim_id=mock_claim.claim_id,
        status=VerificationStatus.VERIFIED,
        reason="numeric_match",
        confidence=0.95,
        evidence_chunk_id="chunk1",
    )


# ============================================================================
# Test Helpers
# ============================================================================


def build_guardrail_result(
    risk_assessment: RiskAssessment,
    should_route_to_audit: bool,
) -> GuardrailPipelineResult:
    """Build a deterministic guardrail pipeline result."""
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
    """Build a deterministic routing decision."""
    return RoutingDecision(
        should_create_audit_record=should_create_audit_record,
        action=action,
        reason=reason,
        audit_priority=audit_priority,
    )


def build_retrieval_result() -> RetrievalResult:
    """Build the standard retrieval result used by graph tests."""
    return RetrievalResult(
        chunk_id="chunk1",
        document_id="doc1",
        text="Revenue was $42.8 billion for fiscal year 2025.",
        score=0.9,
        page_number=3,
        section="Financials",
        document_sha256="abc123",
        retrieval_method="hybrid_rrf",
    )


def build_verified_result() -> VerificationResult:
    """Build a verified claim result for integration tests."""
    return VerificationResult(
        claim_id="claim_1",
        status=VerificationStatus.VERIFIED,
        reason="numeric_match",
        confidence=0.95,
        evidence_chunk_id="chunk1",
    )


def build_verify_side_effect():
    """
    Build a side_effect function for a mocked Verifier.verify().

    claim_generation_node builds each claim's id dynamically
    (question_id + metric + chunk_id), so a static claim_id fixture
    like build_verified_result() will never match a real claim's id.
    answer_generation_node correctly refuses to treat an unmatched
    verification result as backing a claim it doesn't reference, so
    graph-level tests must echo back the *actual* claim id that was
    passed in, exactly as the real Verifier would key its result to
    the claim under test.
    """

    def _verify_side_effect(
        claim: Claim,
        evidence_text: str,
    ) -> VerificationResult:
        # VerificationResult is a frozen dataclass, so we cannot mutate
        # the claim_id on an existing instance — build a fresh one that
        # mirrors build_verified_result() but keyed to the real claim.
        template = build_verified_result()

        return VerificationResult(
            claim_id=claim.claim_id,
            status=template.status,
            reason=template.reason,
            confidence=template.confidence,
            evidence_chunk_id=template.evidence_chunk_id,
        )

    return _verify_side_effect


def build_risk_assessment(
    score: float,
    level: str,
    triggers: list[str],
    action: str,
) -> RiskAssessment:
    """Build a risk assessment."""
    return RiskAssessment(
        risk_score=score,
        risk_level=level,
        triggers=triggers,
        recommended_action=action,
    )


def build_audit_record(
    *,
    risk_assessment: RiskAssessment,
    claim: str,
    verification_status: str,
    verification_reason: str,
) -> AuditRecord:
    """Build a standard audit record."""
    return AuditRecord(
        audit_id="AUD-1",
        timestamp=datetime.now(),
        created_at=datetime.now(),
        user_query="What was revenue in 2025?",
        claim=claim,
        verification_status=verification_status,
        verification_reason=verification_reason,
        risk_level=risk_assessment.risk_level,
        risk_assessment=risk_assessment,
        evidence=[],
        document_id="doc1",
        document_sha256="abc123",
        page_number=3,
        confidence_score=risk_assessment.risk_score,
        risk_score=risk_assessment.risk_score,
        triggers=risk_assessment.triggers,
        verification_results=[],
    )


# ============================================================================
# Query Analysis
# ============================================================================


def test_query_analysis_node_sets_analysis(
    base_state: AgentState,
) -> None:
    """Query analysis should populate structured query information."""
    result = query_analysis_node(base_state)

    assert result.query_analysis is not None
    assert "entities" in result.query_analysis


# ============================================================================
# Retrieval
# ============================================================================


def test_retrieval_node_uses_hybrid_retriever(
    base_state: AgentState,
) -> None:
    """Retrieval node should delegate retrieval to HybridRetriever."""
    
    # Create test chunks instead of empty list
    test_chunks = [
        DocumentChunk(
            chunk_id="test_chunk_1",
            document_id="test_doc",
            text="Revenue was $42.8 billion in 2025.",
            page_number=1,
            section="Financials",
            document_sha256="a" * 64,
        ),
        DocumentChunk(
            chunk_id="test_chunk_2",
            document_id="test_doc",
            text="Net income was $10.2 billion in 2025.",
            page_number=2,
            section="Financials",
            document_sha256="a" * 64,
        ),
    ]
    
    # Mock the document chunks to return test data
    with patch("src.agent.node._load_document_chunks") as mock_load_chunks:
        mock_load_chunks.return_value = test_chunks
        
        with patch("src.agent.node.HybridRetriever") as mock_retriever_cls:
            mock_retriever = mock_retriever_cls.return_value
            mock_retriever.retrieve.return_value = []
            
            # Modify base_state to include query_tasks
            base_state.query_tasks = [
                {
                    "question_id": "q1",
                    "question": "What was revenue in 2025?",
                    "companies": [],
                    "period": "2025",
                }
            ]
            
            # Run the retrieval node
            retrieval_node(base_state)
            
            # Verify retrieve was called at least once
            assert mock_retriever.retrieve.called

# ============================================================================
# Claim Generation
# ============================================================================


def test_claim_generation_node_extracts_numeric_claim(
    base_state: AgentState,
    mock_retrieval_results: list[RetrievalResult],
) -> None:
    """
    Numeric claims should be extracted deterministically from evidence.

    Claim generation must not depend on an LLM to identify a financial
    number that is already explicitly present in retrieved evidence.
    """
    base_state.retrieval_results = mock_retrieval_results

    # Build the same structured query context normally produced by the graph.
    base_state = query_analysis_node(base_state)

    result = claim_generation_node(base_state)

    assert len(result.claims) == 1

    claim = result.claims[0]

    assert claim.subject.lower() == "revenue"
    # Accept the actual format returned by the extraction
    assert claim.value in ["$42.8B", "$42.8 billion"]


def test_claim_generation_node_handles_missing_evidence(
    base_state: AgentState,
) -> None:
    """Missing retrieval evidence should produce no claims."""
    base_state.retrieval_results = []

    result = claim_generation_node(base_state)

    assert result.claims == []
    assert result.raw_llm_output == (
        "The evidence does not contain the requested financial information."
    )


# ============================================================================
# Verification
# ============================================================================


def test_verification_node_calls_verifier(
    base_state: AgentState,
    mock_claim: Claim,
    mock_retrieval_results: list[RetrievalResult],
    mock_verification_result: VerificationResult,
) -> None:
    """Verification node should verify each generated claim against evidence."""
    base_state.claims = [mock_claim]
    base_state.retrieval_results = mock_retrieval_results

    with patch("src.agent.node.Verifier") as mock_verifier_cls:
        mock_verifier = mock_verifier_cls.return_value
        mock_verifier.verify.return_value = mock_verification_result

        result = verification_node(base_state)

    assert result.verification_results == [mock_verification_result]

    mock_verifier.verify.assert_called_once_with(
        mock_claim,
        "Revenue was $42.8 billion for fiscal year 2025.",
    )


# ============================================================================
# Guardrails
# ============================================================================


def test_guardrail_node_sets_risk_and_audit_flag(
    base_state: AgentState,
    mock_retrieval_results: list[RetrievalResult],
    mock_verification_result: VerificationResult,
) -> None:
    """Guardrail node should persist risk assessment and audit routing."""
    base_state.retrieval_results = mock_retrieval_results
    base_state.verification_results = [mock_verification_result]
    base_state.raw_llm_output = "revenue = $42.8B"

    risk = build_risk_assessment(
        score=0.8,
        level="HIGH",
        triggers=["NUMERIC_MISMATCH"],
        action="HUMAN_REVIEW",
    )

    with patch("src.agent.node.GuardrailRunner") as mock_runner_cls:
        mock_runner = mock_runner_cls.return_value
        mock_runner.run_full_pipeline.return_value = build_guardrail_result(
            risk_assessment=risk,
            should_route_to_audit=True,
        )

        result = guardrail_node(base_state)

    assert result.risk_assessment is risk
    assert result.risk_assessment.risk_level == "HIGH"
    assert result.should_route_to_audit is True


# ============================================================================
# Routing
# ============================================================================


def test_routing_node_returns_same_state(
    base_state: AgentState,
) -> None:
    """Routing node currently preserves the existing state object."""
    result = routing_node(base_state)

    assert result is base_state


# ============================================================================
# Answer Generation
# ============================================================================


def test_answer_generation_uses_verified_claims(
    base_state: AgentState,
    mock_retrieval_results: list[RetrievalResult],
    mock_verification_result: VerificationResult,
) -> None:
    """Answer generation should use only verified claims."""
    base_state.retrieval_results = mock_retrieval_results
    base_state.verification_results = [mock_verification_result]

    # Add a claim so the answer generation has something to work with
    base_state.claims = [
        Claim(
            claim_id="claim1",
            claim_type=ClaimType.NUMERIC,
            subject="revenue",
            value="$42.8B",
            unit="billion",
            period="2025",
            source_chunk_id="chunk1",
        )
    ]

    mock_llm = type("MockLLM", (), {})()
    mock_llm.generate = lambda prompt: "The revenue was $42.8B in 2025."

    with patch(
        "src.agent.node.get_llm_client",
        return_value=mock_llm,
    ) as mock_llm_factory, patch(
        "src.agent.node.FinalSafetyValidator"
    ) as mock_validator_cls:
        mock_validator = mock_validator_cls.return_value

        validation_result = type(
            "ValidationResult",
            (),
            {
                "allowed": True,
                "reasons": [],
            },
        )()

        mock_validator.validate.return_value = validation_result

        result = answer_generation_node(base_state)

    assert mock_llm_factory.called
    assert result.final_answer == "The revenue was $42.8B in 2025."


def test_answer_generation_routes_unverified_claims_to_audit(
    base_state: AgentState,
    mock_claim: Claim,
) -> None:
    """
    Claims that cannot be verified must not be automatically answered.

    INCONCLUSIVE is used here because VerificationStatus.UNVERIFIED does
    not exist in the current verification model.
    """
    base_state.claims = [mock_claim]
    base_state.verification_results = [
        VerificationResult(
            claim_id="claim1",
            status=VerificationStatus.INCONCLUSIVE,
            reason="numeric_mismatch",
            confidence=0.2,
            evidence_chunk_id="chunk1",
        )
    ]

    result = answer_generation_node(base_state)

    assert result.final_answer
    assert (
        "human review" in result.final_answer.lower()
        or "cannot" in result.final_answer.lower()
        or "verified" in result.final_answer.lower()
    )


def test_answer_generation_handles_no_claims(
    base_state: AgentState,
) -> None:
    """No verified claims should result in a safe non-answer."""
    base_state.claims = []
    base_state.verification_results = []

    result = answer_generation_node(base_state)

    assert result.final_answer
    assert result.final_answer != ""


# ============================================================================
# Audit
# ============================================================================


def test_audit_node_calls_review_service(
    base_state: AgentState,
    mock_retrieval_results: list[RetrievalResult],
    mock_verification_result: VerificationResult,
) -> None:
    """High-risk states should be handed to ReviewService."""
    base_state.retrieval_results = mock_retrieval_results
    base_state.verification_results = [mock_verification_result]

    risk = build_risk_assessment(
        score=0.8,
        level="HIGH",
        triggers=[],
        action="HUMAN_REVIEW",
    )

    base_state.risk_assessment = risk

    audit_record = build_audit_record(
        risk_assessment=risk,
        claim="revenue = $42.8B",
        verification_status="verified",
        verification_reason="numeric_match",
    )

    review_outcome = ReviewOutcome(
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

    with patch("src.agent.node.ReviewService") as mock_review_cls:
        mock_review = mock_review_cls.return_value
        mock_review.initiate_review.return_value = review_outcome

        result = audit_node(base_state)

    assert result.audit_record == audit_record
    assert "human review" in result.final_answer.lower()
    mock_review.initiate_review.assert_called_once()


# ============================================================================
# Full Graph — Low Risk
# ============================================================================


def test_graph_low_risk_returns_verified_answer() -> None:
    """
    Low-risk verified financial information should reach the user.

    Numeric claim extraction is deterministic. Only final natural-language
    answer generation is supplied by the mocked LLM.
    """
    mock_llm = type("MockLLM", (), {})()
    mock_llm.generate = lambda prompt: (
        "The revenue was $42.8B in 2025."
    )

    risk_low = build_risk_assessment(
        score=0.1,
        level="LOW",
        triggers=[],
        action="AUTO_ANSWER",
    )

    with patch(
        "src.agent.node.HybridRetriever"
    ) as mock_retriever, patch(
        "src.agent.node.get_llm_client",
        return_value=mock_llm,
    ) as mock_llm_factory, patch(
        "src.agent.node.Verifier"
    ) as mock_verifier, patch(
        "src.agent.node.GuardrailRunner"
    ) as mock_guardrail, patch(
        "src.agent.node.ReviewService"
    ) as mock_review, patch(
        "src.agent.node.FinalSafetyValidator"
    ) as mock_validator_cls:
        # ------------------------------------------------------------------
        # Retrieval
        # ------------------------------------------------------------------
        mock_retriever.return_value.retrieve.return_value = [
            build_retrieval_result()
        ]

        # ------------------------------------------------------------------
        # Claim Generation - Force claims to be created
        # ------------------------------------------------------------------
        # The claim_generation_node will extract claims from the evidence

        # ------------------------------------------------------------------
        # Verification
        # ------------------------------------------------------------------
        # claim_generation_node builds claim ids dynamically
        # (question_id + metric + chunk_id), so a static claim_id fixture
        # would never match, and answer_generation_node would (correctly)
        # refuse to treat it as verifying anything. Echo back the real id
        # that was actually passed in.
        mock_verifier.return_value.verify.side_effect = (
            build_verify_side_effect()
        )

        # ------------------------------------------------------------------
        # Guardrails
        # ------------------------------------------------------------------
        mock_guardrail.return_value.run_full_pipeline.return_value = (
            build_guardrail_result(
                risk_assessment=risk_low,
                should_route_to_audit=False,
            )
        )

        # ------------------------------------------------------------------
        # Output safety validation
        # ------------------------------------------------------------------
        mock_validator = mock_validator_cls.return_value

        validation_result = type(
            "ValidationResult",
            (),
            {
                "allowed": True,
                "reasons": [],
            },
        )()

        mock_validator.validate.return_value = validation_result

        # ------------------------------------------------------------------
        # Execute graph
        # ------------------------------------------------------------------
        graph = build_agent_graph()

        result = graph.invoke(
            {"user_query": "What was revenue in 2025?"}
        )

    # The answer should match what the LLM generated
    assert "42.8" in result["final_answer"] or "42.8B" in result["final_answer"]

    assert mock_llm_factory.called
    mock_review.return_value.initiate_review.assert_not_called()


# ============================================================================
# Full Graph — High Risk
# ============================================================================


def test_graph_high_risk_routes_to_audit() -> None:
    """High-risk states should be routed to human review."""
    risk_high = build_risk_assessment(
        score=0.9,
        level="HIGH",
        triggers=["INSUFFICIENT_EVIDENCE"],
        action="HUMAN_REVIEW",
    )

    audit_record = build_audit_record(
        risk_assessment=risk_high,
        claim="No claim extracted",
        verification_status="inconclusive",
        verification_reason="EVIDENCE_MISSING",
    )

    review_outcome = ReviewOutcome(
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

    mock_llm = type("MockLLM", (), {})()
    mock_llm.generate = lambda prompt: ""

    with patch(
        "src.agent.node.HybridRetriever"
    ) as mock_retriever, patch(
        "src.agent.node.get_llm_client",
        return_value=mock_llm,
    ), patch(
        "src.agent.node.Verifier"
    ) as mock_verifier, patch(
        "src.agent.node.GuardrailRunner"
    ) as mock_guardrail, patch(
        "src.agent.node.ReviewService"
    ) as mock_review, patch(
        "src.agent.node.FinalSafetyValidator"
    ) as mock_validator_cls:
        # ------------------------------------------------------------------
        # Retrieval
        # ------------------------------------------------------------------
        mock_retriever.return_value.retrieve.return_value = []

        # ------------------------------------------------------------------
        # Verification
        # ------------------------------------------------------------------
        mock_verifier.return_value.verify.return_value = None

        # ------------------------------------------------------------------
        # Guardrails
        # ------------------------------------------------------------------
        mock_guardrail.return_value.run_full_pipeline.return_value = (
            build_guardrail_result(
                risk_assessment=risk_high,
                should_route_to_audit=True,
            )
        )

        # ------------------------------------------------------------------
        # Review service
        # ------------------------------------------------------------------
        mock_review.return_value.initiate_review.return_value = (
            review_outcome
        )

        # ------------------------------------------------------------------
        # Output validator
        # ------------------------------------------------------------------
        mock_validator = mock_validator_cls.return_value

        validation_result = type(
            "ValidationResult",
            (),
            {
                "allowed": True,
                "reasons": [],
            },
        )()

        mock_validator.validate.return_value = validation_result

        # ------------------------------------------------------------------
        # Execute graph
        # ------------------------------------------------------------------
        graph = build_agent_graph()

        result = graph.invoke(
            {"user_query": "What was revenue in 2025?"}
        )

    assert "human review" in result["final_answer"].lower()
    mock_review.return_value.initiate_review.assert_called_once()


# ============================================================================
# Full Graph — Output Safety
# ============================================================================


def test_graph_output_guard_blocks_unsafe_response() -> None:
    """Unsafe generated answers must be replaced by the safety message."""
    mock_llm = type("MockLLM", (), {})()
    mock_llm.generate = lambda prompt: (
        "The revenue was $42.8B in 2025."
    )

    risk_low = build_risk_assessment(
        score=0.1,
        level="LOW",
        triggers=[],
        action="AUTO_ANSWER",
    )

    with patch(
        "src.agent.node.HybridRetriever"
    ) as mock_retriever, patch(
        "src.agent.node.get_llm_client",
        return_value=mock_llm,
    ), patch(
        "src.agent.node.Verifier"
    ) as mock_verifier, patch(
        "src.agent.node.GuardrailRunner"
    ) as mock_guardrail, patch(
        "src.agent.node.ReviewService"
    ) as mock_review, patch(
        "src.agent.node.FinalSafetyValidator"
    ) as mock_validator_cls:
        # ------------------------------------------------------------------
        # Retrieval
        # ------------------------------------------------------------------
        mock_retriever.return_value.retrieve.return_value = [
            build_retrieval_result()
        ]

        # ------------------------------------------------------------------
        # Verification
        # ------------------------------------------------------------------
        # See test_graph_low_risk_returns_verified_answer for why a
        # static claim_id fixture cannot be used here.
        mock_verifier.return_value.verify.side_effect = (
            build_verify_side_effect()
        )

        # ------------------------------------------------------------------
        # Guardrails
        # ------------------------------------------------------------------
        mock_guardrail.return_value.run_full_pipeline.return_value = (
            build_guardrail_result(
                risk_assessment=risk_low,
                should_route_to_audit=False,
            )
        )

        # ------------------------------------------------------------------
        # Output safety validator — BLOCK
        # ------------------------------------------------------------------
        mock_validator = mock_validator_cls.return_value

        validation_result = type(
            "ValidationResult",
            (),
            {
                "allowed": False,
                "reasons": ["Safety violation detected"],
            },
        )()

        mock_validator.validate.return_value = validation_result

        # ------------------------------------------------------------------
        # Execute graph
        # ------------------------------------------------------------------
        graph = build_agent_graph()

        result = graph.invoke(
            {"user_query": "What was revenue in 2025?"}
        )

    expected_message = (
        "This response could not be validated for safety. "
        "Please consult the original source or contact support."
    )

    assert result["final_answer"] == expected_message
    mock_review.return_value.initiate_review.assert_not_called()
    