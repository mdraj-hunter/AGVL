from fastapi.testclient import TestClient

from api.main import app


def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_run_endpoint():
    c = TestClient(app)
    r = c.post("/v1/run", json={"query": "api smoke"})
    assert r.status_code == 200
    body = r.json()
    assert "context" in body
    assert body["context"]["validated_input"] == "api smoke"
