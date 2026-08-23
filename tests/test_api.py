"""FastAPI contract tests using the deterministic demo model."""

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"] in {"cpu", "cuda"}


def test_predict_returns_valid_schema() -> None:
    response = client.post("/predict", json={"mock_mode": True})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["heavy_rain_probability"] <= 1.0
    assert body["qpe_intensity_mm_hr"] >= 0.0
    assert body["hazard_level"] in {"Low", "Moderate", "High"}
    assert 0.0 <= body["physical_consistency_score"] <= 1.0
    assert body["inference_time_ms"] >= 0.0


def test_explain_returns_valid_schema() -> None:
    response = client.post("/explain", json={"mock_mode": True})
    assert response.status_code == 200
    body = response.json()
    assert body["gradcam_base64_png"]
    assert isinstance(body["channel_attributions"], dict)
    assert len(body["temporal_step_weights"]) == 4
    assert body["diagnostic_narrative"]
    assert body["confidence_tier"] in {"High", "Moderate", "Low / Meteorological Warning"}
