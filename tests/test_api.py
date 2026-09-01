from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": "Compliant Financial RAG & Audit Agent",
        "version": "0.1.0",
        "status": "ok",
    }


def test_query_endpoint_exists() -> None:
    response = client.post(
        "/query",
        json={"user_query": "What was revenue in 2025?"},
    )

    assert response.status_code != 404


@patch("src.api.routes.run_agent")
def test_query_endpoint_returns_agent_result(mock_run_agent) -> None:
    mock_run_agent.return_value = {
        "final_answer": "The revenue was $42.8B in 2025.",
        "final_response_status": "GENERATED",
    }

    response = client.post(
        "/query",
        json={"user_query": "What was revenue in 2025?"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["final_answer"] == "The revenue was $42.8B in 2025."
    assert body["status"] == "GENERATED"

    mock_run_agent.assert_called_once_with(
        "What was revenue in 2025?"
    )