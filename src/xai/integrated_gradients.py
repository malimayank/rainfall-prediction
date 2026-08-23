"""Integrated Gradients for satellite and atmospheric model inputs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

DEFAULT_SATELLITE_CHANNELS = ("TIR-1", "TIR-2", "WV", "MIR", "VIS", "IR")
DEFAULT_ATMOSPHERE_CHANNELS = ("CAPE", "TCWV", "WIND_U", "WIND_V", "OMEGA_500", "RH", "SHEAR", "LCL")


def _score(model: nn.Module, satellite: Tensor, atmosphere: Tensor | None) -> Tensor:
    outputs = model(satellite, atmosphere)
    prediction = outputs["classification"] if isinstance(outputs, dict) else outputs
    return prediction.mean()


def _baseline(value: Tensor, choice: str | Tensor, seasonal_mean: Tensor | None) -> Tensor:
    if isinstance(choice, Tensor):
        return choice.to(device=value.device, dtype=value.dtype)
    if choice == "zero":
        return torch.zeros_like(value)
    if choice == "seasonal":
        if seasonal_mean is None:
            raise ValueError("seasonal_mean is required for a seasonal baseline")
        return seasonal_mean.to(device=value.device, dtype=value.dtype).expand_as(value)
    raise ValueError("baseline must be 'zero', 'seasonal', or a Tensor")


def integrated_gradients(
    model: nn.Module,
    satellite: Tensor,
    atmosphere: Tensor | None = None,
    *,
    satellite_baseline: str | Tensor = "zero",
    atmosphere_baseline: str | Tensor = "zero",
    seasonal_satellite_mean: Tensor | None = None,
    seasonal_atmosphere_mean: Tensor | None = None,
    steps: int = 32,
) -> dict[str, Tensor]:
    """Approximate IG with Gauss-Legendre quadrature over input paths."""
    if steps < 30 or steps > 50:
        raise ValueError("steps must be between 30 and 50")
    model_was_training = model.training
    model.eval()
    sat_base = _baseline(satellite, satellite_baseline, seasonal_satellite_mean)
    atm_base = None if atmosphere is None else _baseline(atmosphere, atmosphere_baseline, seasonal_atmosphere_mean)
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    satellite_total = torch.zeros_like(satellite)
    atmosphere_total = None if atmosphere is None else torch.zeros_like(atmosphere)
    for node, weight in zip(nodes, weights):
        sat_path = (sat_base + float(node) * (satellite - sat_base)).detach().requires_grad_(True)
        atm_path = None if atmosphere is None else (atm_base + float(node) * (atmosphere - atm_base)).detach().requires_grad_(True)
        score = _score(model, sat_path, atm_path)
        gradients = torch.autograd.grad(score, (sat_path, atm_path) if atm_path is not None else (sat_path,), allow_unused=True)
        if gradients[0] is not None:
            satellite_total += float(weight) * gradients[0]
        if atmosphere_total is not None and gradients[1] is not None:
            atmosphere_total += float(weight) * gradients[1]
    result = {"satellite": (satellite - sat_base) * satellite_total}
    if atmosphere_total is not None:
        result["atmosphere"] = (atmosphere - atm_base) * atmosphere_total
    if model_was_training:
        model.train()
    return result


def channel_attribution_percentages(
    attributions: Mapping[str, Tensor],
    satellite_names: tuple[str, ...] = DEFAULT_SATELLITE_CHANNELS,
    atmosphere_names: tuple[str, ...] = DEFAULT_ATMOSPHERE_CHANNELS,
) -> dict[str, float]:
    """Aggregate absolute attribution by channel and normalize to percentages."""
    values: dict[str, float] = {}
    for key, names in (("satellite", satellite_names), ("atmosphere", atmosphere_names)):
        tensor = attributions.get(key)
        if tensor is None:
            continue
        channel_axis = 2 if tensor.ndim == 5 else 1
        magnitudes = tensor.detach().abs().sum(dim=tuple(index for index in range(tensor.ndim) if index != channel_axis))
        for name, magnitude in zip(names, magnitudes.flatten().tolist()):
            values[name] = float(magnitude)
    total = sum(values.values())
    return {name: (value / total * 100.0 if total > 0 else 0.0) for name, value in values.items()}


class IntegratedGradientsEngine:
    def __init__(self, steps: int = 32) -> None:
        self.steps = steps

    def attribute(self, model: nn.Module, satellite: Tensor, atmosphere: Tensor | None = None, **kwargs: Any) -> dict[str, Tensor]:
        return integrated_gradients(model, satellite, atmosphere, steps=self.steps, **kwargs)
