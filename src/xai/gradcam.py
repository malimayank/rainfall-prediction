"""PyTorch-native Grad-CAM, Grad-CAM++, and heatmap overlays."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _normalize(heatmap: Tensor) -> Tensor:
    flat = heatmap.flatten(1)
    minimum = flat.min(dim=1).values[:, None, None]
    maximum = flat.max(dim=1).values[:, None, None]
    return ((heatmap - minimum) / (maximum - minimum).clamp_min(1e-8)).clamp(0, 1)


def _prediction_score(outputs: Any, batch_index: int = 0) -> Tensor:
    prediction = outputs["classification"] if isinstance(outputs, dict) else outputs
    return prediction[batch_index].mean()


class GradCAM:
    """Grad-CAM over a convolutional target layer, aggregated across time."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or self._default_target_layer(model)
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self._hooks = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    @staticmethod
    def _default_target_layer(model: nn.Module) -> nn.Module:
        try:
            return model.satellite_encoder.stages[-1]
        except AttributeError as exc:
            raise ValueError("model must expose satellite_encoder.stages[-1] or provide target_layer") from exc

    def _save_activation(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self.activations = output

    def _save_gradient(self, _module: nn.Module, _inputs: tuple[Tensor, ...], outputs: tuple[Tensor, ...]) -> None:
        self.gradients = outputs[0]

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __call__(self, satellite: Tensor, atmosphere: Tensor | None = None, batch_index: int = 0) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(satellite, atmosphere)
        score = _prediction_score(outputs, batch_index)
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("target layer did not produce activations and gradients")
        activations = self.activations
        gradients = self.gradients
        if satellite.ndim == 5 and activations.shape[0] == satellite.shape[0] * satellite.shape[1]:
            activations = activations.reshape(satellite.shape[0], satellite.shape[1], *activations.shape[1:])[batch_index]
            gradients = gradients.reshape(satellite.shape[0], satellite.shape[1], *gradients.shape[1:])[batch_index]
            activations = activations.mean(dim=0)
            gradients = gradients.mean(dim=0)
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        heatmap = F.relu((weights * activations).sum(dim=0, keepdim=True))
        heatmap = F.interpolate(heatmap[None], size=satellite.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return _normalize(heatmap[None])[0].detach().cpu().numpy().astype("float32")


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ using positive higher-order gradient weighting."""

    def __call__(self, satellite: Tensor, atmosphere: Tensor | None = None, batch_index: int = 0) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(satellite, atmosphere)
        _prediction_score(outputs, batch_index).backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("target layer did not produce activations and gradients")
        activations = self.activations
        gradients = self.gradients
        if satellite.ndim == 5 and activations.shape[0] == satellite.shape[0] * satellite.shape[1]:
            steps = satellite.shape[1]
            activations = activations.reshape(satellite.shape[0], steps, *activations.shape[1:])[batch_index].mean(dim=0)
            gradients = gradients.reshape(satellite.shape[0], steps, *gradients.shape[1:])[batch_index].mean(dim=0)
        positive_gradients = gradients.relu()
        squared = positive_gradients.pow(2)
        cubed = positive_gradients.pow(3)
        denominator = 2.0 * squared + (activations * cubed).sum(dim=(-2, -1), keepdim=True)
        alpha = squared / (denominator + 1e-8)
        weights = (alpha * positive_gradients).sum(dim=(-2, -1), keepdim=True)
        heatmap = F.relu((weights * activations).sum(dim=0, keepdim=True))
        heatmap = F.interpolate(heatmap[None], size=satellite.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return _normalize(heatmap[None])[0].detach().cpu().numpy().astype("float32")


def overlay_heatmap(heatmap: np.ndarray, tir1: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a normalized heatmap over TIR-1, using OpenCV or Matplotlib lazily."""
    heatmap = np.asarray(heatmap, dtype="float32")
    tir1 = np.asarray(tir1, dtype="float32")
    if heatmap.shape != tir1.shape:
        raise ValueError("heatmap and tir1 must have the same spatial shape")
    base = ((tir1 - np.nanmin(tir1)) / (np.nanmax(tir1) - np.nanmin(tir1) + 1e-8) * 255).clip(0, 255).astype("uint8")
    try:
        cv2 = __import__("cv2")
        background = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        colors = cv2.applyColorMap((heatmap * 255).astype("uint8"), cv2.COLORMAP_JET)
        return cv2.addWeighted(background, 1 - alpha, colors, alpha, 0)
    except ImportError:
        import matplotlib.cm as cm
        background = np.repeat(base[..., None], 3, axis=-1) / 255.0
        colors = cm.get_cmap("jet")(heatmap)[..., :3]
        return ((1 - alpha) * background + alpha * colors)
