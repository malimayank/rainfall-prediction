"""Unified rainfall prediction explanation interface."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from src.xai.attention_rollout import temporal_attention_rollout
from src.xai.gradcam import GradCAM
from src.xai.integrated_gradients import IntegratedGradientsEngine, channel_attribution_percentages
from src.xai.physical_validator import MeteorologicalConsistencyChecker


class RainfallExplainer:
    """Run prediction, attribution, temporal, and physical diagnostics together."""

    def __init__(
        self,
        model: nn.Module,
        *,
        ig_steps: int = 30,
        checker: MeteorologicalConsistencyChecker | None = None,
    ) -> None:
        self.model = model
        self.gradcam = GradCAM(model)
        self.integrated_gradients = IntegratedGradientsEngine(steps=ig_steps)
        self.checker = checker or MeteorologicalConsistencyChecker()

    @staticmethod
    def _tensor(value: Any, device: torch.device, add_batch: bool = False) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32)
        tensor = tensor.to(device=device, dtype=torch.float32)
        return tensor.unsqueeze(0) if add_batch else tensor

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if isinstance(value, Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def explain(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        satellite_value = sample["satellite"]
        satellite_value = satellite_value if isinstance(satellite_value, Tensor) else torch.as_tensor(satellite_value)
        satellite = self._tensor(
            satellite_value,
            next(self.model.parameters()).device,
            add_batch=satellite_value.ndim == 4,
        )
        atmosphere_value = sample.get("atmosphere")
        if atmosphere_value is None:
            atmosphere = None
        else:
            atmosphere_tensor = atmosphere_value if isinstance(atmosphere_value, Tensor) else torch.as_tensor(atmosphere_value)
            atmosphere = self._tensor(atmosphere_tensor, satellite.device, add_batch=atmosphere_tensor.ndim == 3)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(satellite, atmosphere)
        probability = float(outputs["classification"].mean().item())
        qpe = float(outputs["qpe"].mean().item())
        heatmap = self.gradcam(satellite, atmosphere)
        attributions = self.integrated_gradients.attribute(self.model, satellite, atmosphere)
        percentages = channel_attribution_percentages(attributions)
        temporal = temporal_attention_rollout(self.model)
        tir1 = self._array(sample.get("tir1", satellite[0, -1, 0]))
        btd = sample.get("split_window_btd")
        wv = sample.get("wv")
        if btd is None and wv is None and satellite.shape[2] >= 3:
            wv = satellite[0, -1, 2].detach().cpu().numpy()
        risk = outputs["classification"][0, 0].detach().cpu().numpy()
        validation = self.checker.validate(
            heatmap,
            tir1,
            None if btd is None else self._array(btd),
            None if wv is None else self._array(wv),
            risk,
            None if sample.get("cape") is None else self._array(sample["cape"]),
            None if sample.get("omega_500") is None else self._array(sample["omega_500"]),
        )
        diagnostic_text = self.checker.synthesize_explanation(validation)
        if was_training:
            self.model.train()
        return {
            "prediction": {"heavy_rain_prob": probability, "qpe_mm_hr": qpe},
            "gradcam_heatmap": heatmap,
            "channel_attributions": percentages,
            "temporal_weights": temporal["weights"],
            "physics_validation": validation,
            "diagnostic_text": diagnostic_text,
        }

    __call__ = explain
