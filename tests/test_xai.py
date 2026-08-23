"""Tests for model attribution and physical consistency diagnostics."""

import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.models.temporal_attention import SpatioTemporalCrossAttention
from src.xai.attention_rollout import temporal_attention_rollout
from src.xai.explainer import RainfallExplainer
from src.xai.gradcam import GradCAM
from src.xai.integrated_gradients import integrated_gradients
from src.xai.physical_validator import MeteorologicalConsistencyChecker


class TinyExplainableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.satellite_encoder = nn.Module()
        self.satellite_encoder.stages = nn.ModuleList([nn.Conv2d(2, 8, 3, stride=8, padding=1)])
        self.temporal_attention = SpatioTemporalCrossAttention(8, num_heads=2, max_sequence_length=4)
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, satellite: torch.Tensor, atmosphere: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, steps, channels, height, width = satellite.shape
        encoded = self.satellite_encoder.stages[-1](satellite.reshape(batch * steps, channels, height, width))
        _, features, small_height, small_width = encoded.shape
        sequence = encoded.reshape(batch, steps, features, small_height, small_width)
        attended = self.temporal_attention(sequence)
        logits = F.interpolate(self.head(attended), size=(height, width), mode="bilinear", align_corners=False)
        return {"classification": torch.sigmoid(logits), "qpe": F.softplus(logits)}


def test_gradcam_heatmap_is_normalized() -> None:
    model = TinyExplainableModel()
    heatmap = GradCAM(model)(torch.randn(1, 4, 2, 128, 128))
    assert heatmap.shape == (128, 128)
    assert np.isfinite(heatmap).all()
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


def test_integrated_gradients_are_finite() -> None:
    model = TinyExplainableModel()
    satellite = torch.randn(1, 4, 2, 16, 16)
    attributions = integrated_gradients(model, satellite, steps=30)
    assert attributions["satellite"].shape == satellite.shape
    assert torch.isfinite(attributions["satellite"]).all()


def test_physical_checker_scores_dummy_inputs() -> None:
    size = 16
    activation = np.zeros((size, size), dtype="float32")
    activation[4:12, 4:12] = 1.0
    tir1 = 270.0 - activation * 20.0
    checker = MeteorologicalConsistencyChecker()
    result = checker.validate(
        activation, tir1, wv=tir1 + 1.0,
        cape=np.full((size, size), 1500.0),
        risk_probability=activation,
    )
    assert 0.0 <= result["score"] <= 1.0
    assert result["confidence"] in {"High", "Moderate", "Low / Meteorological Warning"}
    json.loads(checker.synthesize_explanation(result))


def test_master_explainer_returns_expected_schema() -> None:
    model = TinyExplainableModel()
    satellite = torch.randn(4, 2, 128, 128)
    result = RainfallExplainer(model, ig_steps=30)({"satellite": satellite})
    assert set(result) == {
        "prediction", "gradcam_heatmap", "channel_attributions",
        "temporal_weights", "physics_validation", "diagnostic_text",
    }
    assert result["gradcam_heatmap"].shape == (128, 128)
    assert len(result["temporal_weights"]) == 4
    json.loads(result["diagnostic_text"])
