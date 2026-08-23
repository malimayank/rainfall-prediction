"""Asynchronous FastAPI service for prediction and explanation demos."""

from __future__ import annotations

import base64
import struct
import time
import zlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from torch import Tensor, nn
import torch.nn.functional as F

from api.schemas import DemoSampleResponse, ExplainResponse, PredictionRequest, PredictionResponse
from src.xai.explainer import RainfallExplainer


class DemoInferenceModel(nn.Module):
    """Small deterministic model used until a trained checkpoint is configured."""

    def __init__(self) -> None:
        super().__init__()
        self.satellite_encoder = nn.Module()
        self.satellite_encoder.stages = nn.ModuleList([nn.Conv2d(6, 16, 3, stride=8, padding=1)])
        from src.models.temporal_attention import SpatioTemporalCrossAttention
        self.temporal_attention = SpatioTemporalCrossAttention(16, num_heads=4, max_sequence_length=4)
        self.classification_head = nn.Conv2d(16, 1, 1)
        self.qpe_head = nn.Conv2d(16, 1, 1)

    def forward(self, satellite: Tensor, atmosphere: Tensor | None = None) -> dict[str, Tensor]:
        del atmosphere
        batch, steps, channels, height, width = satellite.shape
        encoded = self.satellite_encoder.stages[-1](satellite.reshape(batch * steps, channels, height, width))
        _, channels, small_height, small_width = encoded.shape
        sequence = encoded.reshape(batch, steps, channels, small_height, small_width)
        attended = self.temporal_attention(sequence)
        classification = F.interpolate(self.classification_head(attended), size=(height, width), mode="bilinear", align_corners=False)
        qpe = F.interpolate(self.qpe_head(attended), size=(height, width), mode="bilinear", align_corners=False)
        return {"classification": torch.sigmoid(classification), "qpe": F.softplus(qpe)}


torch.manual_seed(7)
model = DemoInferenceModel().eval()
explainer = RainfallExplainer(model, ig_steps=30)
app = FastAPI(title="Explainable Heavy Rainfall Prediction API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse("frontend/index.html")


def _demo_tensors() -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(17)
    satellite = torch.randn(1, 4, 6, 128, 128, generator=generator) * 0.5
    satellite[:, :, 0] += 250.0
    atmosphere = torch.randn(1, 8, 128, 128, generator=generator)
    atmosphere[:, 0] = 1200.0 + atmosphere[:, 0] * 200.0
    atmosphere[:, 1] = 45.0 + atmosphere[:, 1] * 8.0
    return satellite, atmosphere


def _request_tensors(request: PredictionRequest) -> tuple[Tensor, Tensor]:
    if request.mock_mode or request.sequence_inputs is None:
        return _demo_tensors()
    satellite = torch.as_tensor(request.sequence_inputs, dtype=torch.float32)
    if satellite.ndim == 4:
        satellite = satellite.unsqueeze(0)
    if satellite.shape != (1, 4, 6, 128, 128):
        raise HTTPException(status_code=422, detail="sequence_inputs must be [4, 6, 128, 128]")
    if request.atmosphere_inputs is None:
        atmosphere = torch.zeros(1, 8, 128, 128)
    else:
        atmosphere = torch.as_tensor(request.atmosphere_inputs, dtype=torch.float32)
        if atmosphere.ndim == 3:
            atmosphere = atmosphere.unsqueeze(0)
        if atmosphere.shape != (1, 8, 128, 128):
            raise HTTPException(status_code=422, detail="atmosphere_inputs must be [8, 128, 128]")
    return satellite, atmosphere


def _hazard(probability: float) -> str:
    return "High" if probability >= 0.7 else "Moderate" if probability >= 0.4 else "Low"


def _png_base64(heatmap: np.ndarray) -> str:
    values = np.asarray(heatmap, dtype="float32").clip(0.0, 1.0)
    red = (255 * values).astype("uint8")
    blue = (255 * (1.0 - values)).astype("uint8")
    green = (255 * (1.0 - np.abs(values - 0.5) * 2.0)).astype("uint8")
    rgb = np.stack([red, green, blue], axis=-1)
    raw = b"".join(b"\x00" + row.tobytes() for row in rgb)
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", rgb.shape[1], rgb.shape[0], 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.get("/demo-sample", response_model=DemoSampleResponse)
async def demo_sample() -> DemoSampleResponse:
    return DemoSampleResponse(
        bounding_box=[65.0, 5.0, 95.0, 38.0],
        timestamp=datetime.now(timezone.utc),
        sequence_shape=[4, 6, 128, 128],
        atmosphere_shape=[8, 128, 128],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    satellite, atmosphere = _request_tensors(request)
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model(satellite, atmosphere)
    elapsed = (time.perf_counter() - started) * 1000.0
    probability = float(outputs["classification"].mean())
    qpe = float(outputs["qpe"].mean())
    return PredictionResponse(
        heavy_rain_probability=probability,
        qpe_intensity_mm_hr=qpe,
        hazard_level=_hazard(probability),
        physical_consistency_score=0.5,
        inference_time_ms=elapsed,
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: PredictionRequest) -> ExplainResponse:
    satellite, atmosphere = _request_tensors(request)
    result = explainer({"satellite": satellite, "atmosphere": atmosphere})
    physics = result["physics_validation"]
    return ExplainResponse(
        gradcam_base64_png=_png_base64(result["gradcam_heatmap"]),
        channel_attributions=result["channel_attributions"],
        temporal_step_weights=result["temporal_weights"],
        diagnostic_narrative=result["diagnostic_text"],
        confidence_tier=physics["confidence"],
    )
