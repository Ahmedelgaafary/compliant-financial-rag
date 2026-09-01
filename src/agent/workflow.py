"""Agent workflow entry point."""

from src.agent.graph import build_agent_graph
from src.agent.state import AgentState


def run_agent(user_query: str) -> AgentState:
    """
    Run the complete agent workflow with a user query.
    
    Args:
        user_query: The user's financial question
        
    Returns:
        AgentState: The final state with all results
    """
    graph = build_agent_graph()
    initial_state = AgentState(user_query=user_query)
    result = graph.invoke(initial_state)
    
    # If result is a dict (LangGraph default), convert to AgentState
    if isinstance(result, dict):
        return AgentState(**result)
    
    return result


def run_agent_raw(user_query: str) -> dict:
    """
    Run the agent and return raw dictionary results.
    
    Args:
        user_query: The user's financial question
        
    Returns:
        dict: Raw results from the graph
    """
    graph = build_agent_graph()
    initial_state = AgentState(user_query=user_query)
    return graph.invoke(initial_state)