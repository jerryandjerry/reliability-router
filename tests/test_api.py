from fastapi.testclient import TestClient

from reliability_router.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_route_endpoint() -> None:
    response = client.post("/v1/route", json={"query": "What is the latest weather today?"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "DEEP"
    assert "FRESHNESS_REQUIRED" in body["reason_codes"]


def test_answer_endpoint() -> None:
    response = client.post("/v1/answer", json={"query": "What is 2 + 2?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "4"
