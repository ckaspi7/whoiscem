"""api.py (Phase 4.1): headless /chat and /healthz, with auth, rate limiting,
and a daily spend cap enforced at the boundary rather than inside chatbot.py's
Streamlit UI code. Fully mocked: chatbot.create_assistant (called once by the
lifespan) and retrieval.backends.create_qdrant_client never touch a real
index, API, or Qdrant instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import api
from rate_limit import RateLimiter


@pytest.fixture(autouse=True)
def _fresh_rate_limiter():
    """api._rate_limiter is a module-level singleton — without resetting it,
    one test's calls would count against the next test's limit."""
    api._rate_limiter = RateLimiter(limit=api.DEFAULT_RATE_LIMIT_PER_MINUTE, window_seconds=60)
    yield


@pytest.fixture(autouse=True)
def _clear_api_key_and_spend_cap_env(monkeypatch):
    """Tests control these explicitly per case; a real .env value leaking in
    would make auth/spend-cap tests depend on the machine running them."""
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("DAILY_SPEND_CAP_USD", raising=False)
    monkeypatch.delenv("RATE_LIMIT_PER_MINUTE", raising=False)


def _mock_graph(answer: str = "Cem works at TELUS.", route: str = "resume", **state_overrides):
    graph = MagicMock()

    def _invoke(state):
        result = {
            **state,
            "messages": [*state["messages"], AIMessage(content=answer)],
            "route": route,
            "faithfulness_score": 5,
            "token_usage": {
                "route_query": {"input": 50, "output": 2},
                "generate_response": {"input": 200, "output": 30},
                "check_faithfulness": {"input": 300, "output": 10},
            },
        }
        result.update(state_overrides)
        return result

    graph.invoke.side_effect = _invoke
    return graph


@pytest.fixture
def client():
    with patch("chatbot.create_assistant", return_value=_mock_graph()):
        with TestClient(api.app) as test_client:
            yield test_client


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------


def test_healthz_reports_healthy_when_qdrant_and_the_api_key_are_present(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("api.create_qdrant_client") as qdrant_cls:
        qdrant_cls.return_value.get_collections.return_value = None
        response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["healthy"] is True
    assert body["checks"]["qdrant"] is True
    assert body["checks"]["openai_api_key"] is True


def test_healthz_reports_unhealthy_when_qdrant_is_unreachable(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("api.create_qdrant_client", side_effect=ConnectionError("refused")):
        response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["healthy"] is False
    assert response.json()["checks"]["qdrant"] is False


def test_healthz_reports_unhealthy_with_no_api_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("api.create_qdrant_client") as qdrant_cls:
        qdrant_cls.return_value.get_collections.return_value = None
        response = client.get("/healthz")

    assert response.json()["healthy"] is False
    assert response.json()["checks"]["openai_api_key"] is False


def test_healthz_reports_redis_as_informational_not_a_health_failure(client, monkeypatch):
    """Redis absence degrades a feature elsewhere in this project; it must
    not make /healthz report unhealthy."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("REDIS_URL", raising=False)
    with patch("api.create_qdrant_client") as qdrant_cls:
        qdrant_cls.return_value.get_collections.return_value = None
        response = client.get("/healthz")

    assert response.json()["healthy"] is True
    assert "not configured" in response.json()["checks"]["redis"]


# ---------------------------------------------------------------------------
# /chat — the happy path and its response shape
# ---------------------------------------------------------------------------


def test_chat_returns_the_answer_route_and_a_real_cost(client):
    response = client.post("/chat", json={"message": "Where does Cem work?"})

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Cem works at TELUS."
    assert body["route"] == "resume"
    assert body["faithfulness_score"] == 5
    assert body["cost_usd"] > 0


def test_chat_passes_history_through_as_real_messages():
    graph = _mock_graph()
    with patch("chatbot.create_assistant", return_value=graph), TestClient(api.app) as client:
        client.post(
            "/chat",
            json={
                "message": "How long was that?",
                "history": [
                    {"role": "human", "content": "What startup did Cem co-found?"},
                    {"role": "ai", "content": "NeoWise."},
                ],
            },
        )

    sent_state = graph.invoke.call_args.args[0]
    assert [m.type for m in sent_state["messages"]] == ["human", "ai", "human"]
    assert sent_state["messages"][-1].content == "How long was that?"


def test_an_oversized_message_is_rejected_by_the_request_schema(client):
    response = client.post("/chat", json={"message": "x" * (api.MAX_INPUT_CHARS + 1)})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_no_api_key_configured_allows_requests_through(client):
    response = client.post("/chat", json={"message": "Where does Cem work?"})
    assert response.status_code == 200


def test_a_configured_api_key_rejects_a_missing_or_wrong_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "the-real-key")
    response = client.post("/chat", json={"message": "Where does Cem work?"})
    assert response.status_code == 401

    response = client.post(
        "/chat", json={"message": "Where does Cem work?"}, headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401


def test_a_configured_api_key_accepts_the_correct_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "the-real-key")
    response = client.post(
        "/chat", json={"message": "Where does Cem work?"}, headers={"X-API-Key": "the-real-key"}
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_requests_beyond_the_limit_are_rejected_with_retry_after(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    api._rate_limiter = RateLimiter(limit=1, window_seconds=60)

    first = client.post("/chat", json={"message": "Where does Cem work?"})
    second = client.post("/chat", json={"message": "Where does Cem work?"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Retry-After" in second.headers


def test_different_api_keys_have_independent_rate_limit_buckets(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    api._rate_limiter = RateLimiter(limit=1, window_seconds=60)

    first = client.post("/chat", json={"message": "Where does Cem work?"}, headers={"X-API-Key": "client-a"})
    second = client.post("/chat", json={"message": "Where does Cem work?"}, headers={"X-API-Key": "client-b"})

    assert first.status_code == 200
    assert second.status_code == 200


# ---------------------------------------------------------------------------
# Daily spend cap
# ---------------------------------------------------------------------------


def test_requests_are_rejected_once_the_daily_spend_cap_is_reached(client, monkeypatch):
    monkeypatch.setenv("DAILY_SPEND_CAP_USD", "0.0000001")  # any real call exceeds this

    first = client.post("/chat", json={"message": "Where does Cem work?"})
    second = client.post("/chat", json={"message": "Where does Cem work?"})

    assert first.status_code == 200  # the call that pushes spend over the cap still completes
    assert second.status_code == 402


def test_spend_accumulates_across_requests_within_the_cap(client, monkeypatch):
    monkeypatch.setenv("DAILY_SPEND_CAP_USD", "1000")  # comfortably above what two mocked calls cost

    first = client.post("/chat", json={"message": "Where does Cem work?"})
    second = client.post("/chat", json={"message": "What did Cem study?"})

    assert first.status_code == 200
    assert second.status_code == 200
