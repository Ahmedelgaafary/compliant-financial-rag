"""Agent workflow graph builder."""

from langgraph.graph import END, StateGraph

from src.agent.node import (
    answer_generation_node,
    audit_node,
    claim_generation_node,
    financial_intelligence_node,
    guardrail_node,
    output_guard_node,
    query_analysis_node,
    retrieval_node,
    routing_node,
    verification_node,
)
from src.agent.state import AgentState


def build_agent_graph():
    """
    Build the complete compliant financial RAG workflow.

    Flow:

        Query Analysis
            ↓
        Multi-question / multi-company Retrieval
            ↓
        Deterministic Claim Generation
            ↓
        Scoped Verification
            ↓
        Financial Intelligence
            ↓
        Guardrails
            ↓
        Routing
          ↙   ↘
       Audit  Answer Generation
                 ↓
             Output Guard
    """

    workflow = StateGraph(
        AgentState
    )

    workflow.add_node(
        "query_analysis",
        query_analysis_node,
    )

    workflow.add_node(
        "retrieval",
        retrieval_node,
    )

    workflow.add_node(
        "claim_generation",
        claim_generation_node,
    )

    workflow.add_node(
        "verification",
        verification_node,
    )

    workflow.add_node(
        "financial_intelligence",
        financial_intelligence_node,
    )

    workflow.add_node(
        "guardrail",
        guardrail_node,
    )

    workflow.add_node(
        "routing",
        routing_node,
    )

    workflow.add_node(
        "answer_generation",
        answer_generation_node,
    )

    workflow.add_node(
        "output_guard",
        output_guard_node,
    )

    workflow.add_node(
        "audit",
        audit_node,
    )

    workflow.set_entry_point(
        "query_analysis"
    )

    workflow.add_edge(
        "query_analysis",
        "retrieval",
    )

    workflow.add_edge(
        "retrieval",
        "claim_generation",
    )

    workflow.add_edge(
        "claim_generation",
        "verification",
    )

    workflow.add_edge(
        "verification",
        "financial_intelligence",
    )

    workflow.add_edge(
        "financial_intelligence",
        "guardrail",
    )

    workflow.add_edge(
        "guardrail",
        "routing",
    )

    def should_route_to_audit(
        state: AgentState,
    ) -> str:
        if state.should_route_to_audit:
            return "audit"

        return "answer_generation"

    workflow.add_conditional_edges(
        "routing",
        should_route_to_audit,
        {
            "audit": "audit",
            "answer_generation": (
                "answer_generation"
            ),
        },
    )

    workflow.add_edge(
        "audit",
        END,
    )

    workflow.add_edge(
        "answer_generation",
        "output_guard",
    )

    workflow.add_edge(
        "output_guard",
        END,
    )

    return workflow.compile()


def run_agent(
    user_query: str,
) -> AgentState:
    """Run the agent with a user query."""

    graph = build_agent_graph()

    initial_state = AgentState(
        user_query=user_query
    )

    return graph.invoke(
        initial_state
    )