from src.agent.node import verification_node
from src.agent.state import AgentState
from src.retrieval.models import RetrievalResult
from src.verification.models import (
    Claim,
    ClaimType,
    VerificationStatus,
)


def make_retrieval_result(
    text: str,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-001",
        document_id="doc-001",
        text=text,
        score=1.0,
        page_number=5,
        section="Revenue",
        document_sha256="abc123",
        retrieval_method="hybrid",
    )


def make_numeric_claim(
    value: str = "$42.8",
) -> Claim:
    return Claim(
        claim_id="claim-001",
        claim_type=ClaimType.NUMERIC,
        subject="revenue",
        value=value,
        unit="billion",
        period=None,
        source_chunk_id="chunk-001",
    )


def test_verification_node_uses_central_verification_engine() -> None:
    state = AgentState(
        user_query="What was the revenue?",
        retrieval_results=[
            make_retrieval_result(
                "Revenue was $42.8 billion."
            )
        ],
        claims=[
            make_numeric_claim()
        ],
    )

    result = verification_node(state)

    assert len(result.verification_results) == 1

    verification_result = result.verification_results[0]

    assert verification_result.claim_id == "claim-001"
    assert verification_result.status == VerificationStatus.VERIFIED
    assert verification_result.evidence_chunk_id == "chunk-001"


def test_verification_node_rejects_contradicted_claim() -> None:
    state = AgentState(
        user_query="What was the revenue?",
        retrieval_results=[
            make_retrieval_result(
                "Revenue was not $42.8 billion."
            )
        ],
        claims=[
            make_numeric_claim()
        ],
    )

    result = verification_node(state)

    assert len(result.verification_results) == 1

    verification_result = result.verification_results[0]

    assert verification_result.status == VerificationStatus.REJECTED


def test_verification_node_handles_multiple_claims() -> None:
    state = AgentState(
        user_query="What were the reported revenues?",
        retrieval_results=[
            make_retrieval_result(
                "Revenue was $42.8 billion."
            )
        ],
        claims=[
            make_numeric_claim("$42.8"),
            make_numeric_claim("$99.9"),
        ],
    )

    result = verification_node(state)

    assert len(result.verification_results) == 2

    assert (
        result.verification_results[0].status
        == VerificationStatus.VERIFIED
    )

    assert (
        result.verification_results[1].status
        == VerificationStatus.REJECTED
    )