
from src.agent.graph import build_agent_graph
from src.agent.state import AgentState


def run_agent(user_query: str) -> AgentState:
    """Run the full agent pipeline and return the final state."""
    graph = build_agent_graph()
    initial_state = AgentState(user_query=user_query)
    final_state = graph.invoke(initial_state)
    return final_state