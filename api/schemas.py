"""Pydantic request and response contracts for the rainfall API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    bounding_box: list[float] = Field(default=[65.0, 5.0, 95.0, 38.0], min_length=4, max_length=4)
    timestamp: datetime | None = None
    sequence_inputs: list[Any] | None = None
    atmosphere_inputs: list[Any] | None = None
    mock_mode: bool = True


class PredictionResponse(BaseModel):
    heavy_rain_probability: float = Field(ge=0.0, le=1.0)
    qpe_intensity_mm_hr: float = Field(ge=0.0)
    hazard_level: str
    physical_consistency_score: float = Field(ge=0.0, le=1.0)
    inference_time_ms: float = Field(ge=0.0)


class ExplainResponse(BaseModel):
    gradcam_base64_png: str
    channel_attributions: dict[str, float]
    temporal_step_weights: list[float]
    diagnostic_narrative: str
    confidence_tier: str


class DemoSampleResponse(BaseModel):
    bounding_box: list[float]
    timestamp: datetime
    sequence_shape: list[int]
    atmosphere_shape: list[int]
    mock_mode: bool = True
