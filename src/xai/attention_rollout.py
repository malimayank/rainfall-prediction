"""Temporal attention rollout and historical influence summaries."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

DEFAULT_TIME_LABELS = ("t-90", "t-60", "t-30", "t")


def extract_temporal_rollout(model: nn.Module) -> np.ndarray:
    """Read the latest per-pixel rollout retained by the attention module."""
    attention = getattr(getattr(model, "temporal_attention", None), "get_attention_rollout", lambda: None)()
    if attention is None:
        raise RuntimeError("run a model forward pass before extracting attention rollout")
    if isinstance(attention, Tensor):
        attention = attention.detach().float().cpu().numpy()
    return np.asarray(attention, dtype="float32")


def temporal_attention_rollout(
    model: nn.Module,
    time_labels: Sequence[str] = DEFAULT_TIME_LABELS,
) -> dict[str, Any]:
    """Average [B,H,W,K] rollout over batch and space and identify its maximum."""
    rollout = extract_temporal_rollout(model)
    if rollout.ndim == 4:
        weights = rollout.mean(axis=(0, 1, 2))
    elif rollout.ndim == 3:
        weights = rollout.mean(axis=(0, 1))
    else:
        raise ValueError("attention rollout must have shape [batch, height, width, time]")
    weights = weights / (weights.sum() + 1e-8)
    labels = list(time_labels)[: len(weights)]
    return {
        "weights": [float(value) for value in weights],
        "labels": labels,
        "most_influential_time": labels[int(np.argmax(weights))],
    }
