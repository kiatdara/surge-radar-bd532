"""4.2 Interface tests (same scenarios as the Postman collection)."""
import copy

import pytest
from fastapi.testclient import TestClient

from api import app as app_module
from api.app import app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app_module.FEEDBACK_PATH = tmp_path_factory.mktemp("fb") / "feedback.jsonl"
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample(client):
    pid = client.get("/api/v1/products/top?k=1").json()["items"][0]["product_id"]
    return client.get(f"/api/v1/samples/{pid}").json()


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["model_version"].startswith("surge-")


def test_scenario1_predict_real_product(client, sample):
    r = client.post("/api/v1/predict", json=sample)
    assert r.status_code == 200
    p = r.json()["prediction"]
    assert r.json()["request_id"] == sample["request_id"]
    assert 0 <= p["surge_probability"] <= 1 and p["label"] in {"hot", "watch", "skip"}
    assert len(p["reasons"]) == 3 and r.json()["latency_ms"]["total"] > 0


def test_predict_matches_stored_snapshot(client, sample):
    a = client.post("/api/v1/predict", json=sample).json()["prediction"]["surge_probability"]
    b = client.get(f"/api/v1/products/{sample['product']['product_id']}/prediction").json()
    assert a == pytest.approx(b["surge_probability"], abs=1e-4)


def test_scenario2_top_list_sorted_and_fair(client):
    r = client.get("/api/v1/products/top?k=10")
    ps = [x["surge_probability"] for x in r.json()["items"]]
    assert r.status_code == 200 and len(ps) == 10 and ps == sorted(ps, reverse=True)
    f = client.get("/api/v1/products/top?k=10&fair=true").json()
    assert f["fair_rerank"] is True and len(f["items"]) == 10


def test_thai_reasons(client, sample):
    s = copy.deepcopy(sample); s["options"]["language"] = "th"
    txt = client.post("/api/v1/predict", json=s).json()["prediction"]["reasons"][0]["text"]
    assert "โอกาส" in txt


def test_scenario3_invalid_value_422(client, sample):
    s = copy.deepcopy(sample); s["product"]["product_rating"] = 9
    r = client.post("/api/v1/predict", json=s)
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_VALUE"
    assert r.json()["error"]["field"] == "product.product_rating"


def test_missing_field_400(client, sample):
    s = copy.deepcopy(sample); del s["product"]
    r = client.post("/api/v1/predict", json=s)
    assert r.status_code == 400 and r.json()["error"]["code"] == "MISSING_FIELD"


def test_short_history_422(client, sample):
    s = copy.deepcopy(sample); s["history"] = s["history"][-10:]
    r = client.post("/api/v1/predict", json=s)
    assert r.status_code == 422 and r.json()["error"]["code"] == "INSUFFICIENT_HISTORY"


def test_unsorted_history_422(client, sample):
    s = copy.deepcopy(sample); s["history"] = s["history"][::-1]
    assert client.post("/api/v1/predict", json=s).status_code == 422


def test_bad_json_400_and_wrong_media_415(client):
    r = client.post("/api/v1/predict", content="{bad", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_JSON"
    r = client.post("/api/v1/predict", content="x", headers={"content-type": "text/plain"})
    assert r.status_code == 415


def test_unknown_product_404(client):
    r = client.get("/api/v1/products/999/prediction")
    assert r.status_code == 404 and r.json()["error"]["code"] == "PRODUCT_NOT_FOUND"


def test_feedback_201(client, sample):
    r = client.post("/api/v1/feedback", json={"product_id": sample["product"]["product_id"],
                                              "prediction_date": "2026-09-16", "label": "useful"})
    assert r.status_code == 201 and r.json()["status"] == "recorded"
    assert app_module.FEEDBACK_PATH.read_text().count("useful") == 1


def test_rate_limit_429(client, monkeypatch):
    monkeypatch.setattr(app_module, "RATE_LIMIT_PER_MIN", 1)
    app_module._hits.clear()
    client.get("/api/v1/health")
    r = client.get("/api/v1/health")
    assert r.status_code == 429 and "Retry-After" in r.headers
    app_module._hits.clear()


def test_model_not_ready_503(client, sample, monkeypatch):
    monkeypatch.setitem(app_module.STATE, "ready", False)
    r = client.post("/api/v1/predict", json=sample)
    assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_READY"
