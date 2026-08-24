import pytest
from httpx import AsyncClient, ASGITransport
from api.app import app

@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

@pytest.mark.asyncio
async def test_root_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_predict_endpoint_get():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/predict")
        assert response.status_code == 200
        data = response.json()
        assert "max_probability" in data
        assert "peak_qpe_mmh" in data
        assert "alert_level" in data
        assert "channel_attributions" in data
        assert "physical_validation" in data

@pytest.mark.asyncio
async def test_predict_endpoint_post():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/predict", json={})
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["max_probability"] <= 1.0
        assert data["peak_qpe_mmh"] >= 0.0
