# src/agent/graph.py
from langgraph.graph import END, StateGraph

from src.agent.node import (
    answer_generation_node,
    audit_node,
    claim_generation_node,
    guardrail_node,
    output_guard_node,
    query_analysis_node,
    retrieval_node,
    routing_node,
    verification_node,
)
from src.agent.state import AgentState


def build_agent_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("claim_generation", claim_generation_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("guardrails", guardrail_node)
    workflow.add_node("routing", routing_node)
    workflow.add_node("answer_generation", answer_generation_node)
    workflow.add_node("output_guard", output_guard_node)
    workflow.add_node("audit", audit_node)

    # Define edges
    workflow.set_entry_point("query_analysis")
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "claim_generation")
    workflow.add_edge("claim_generation", "verification")
    workflow.add_edge("verification", "guardrails")
    workflow.add_edge("guardrails", "routing")

    # Conditional routing based on should_route_to_audit
    def route_after_routing(state: AgentState) -> str:
        if state.should_route_to_audit:
            return "audit"
        else:
            return "answer_generation"

    workflow.add_conditional_edges(
        "routing",
        route_after_routing,
        {
            "audit": "audit",
            "answer_generation": "answer_generation",
        }
    )

    # After answer generation, run output guard, then END
    workflow.add_edge("answer_generation", "output_guard")
    workflow.add_edge("output_guard", END)

    # Audit ends the workflow (after creating audit record)
    workflow.add_edge("audit", END)

    return workflow.compile()