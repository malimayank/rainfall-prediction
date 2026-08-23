"""Classical and neural baseline models for comparison experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TabularBaselineRunner:
    """Fit a point-wise tabular model with 3x3 neighborhood statistics."""

    def __init__(self, model_type: str = "sklearn", task: str = "classification", random_state: int = 42) -> None:
        if model_type not in {"sklearn", "xgboost", "lightgbm"}:
            raise ValueError("model_type must be sklearn, xgboost, or lightgbm")
        if task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        self.model_type = model_type
        self.task = task
        self.random_state = random_state
        self.model: Any | None = None

    @staticmethod
    def extract_features(frames: np.ndarray) -> np.ndarray:
        """Flatten [N,C,H,W] frames into pixels plus local mean/std features."""
        frames = np.asarray(frames, dtype="float32")
        if frames.ndim != 4:
            raise ValueError("frames must have shape [samples, channels, height, width]")
        padded = np.pad(frames, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
        neighborhoods = []
        for row_offset in range(3):
            for col_offset in range(3):
                neighborhoods.append(padded[..., row_offset:row_offset + frames.shape[2], col_offset:col_offset + frames.shape[3]])
        neighborhood = np.stack(neighborhoods, axis=2)
        local_mean = neighborhood.mean(axis=2)
        local_std = neighborhood.std(axis=2)
        return np.concatenate(
            [frames.transpose(0, 2, 3, 1), local_mean.transpose(0, 2, 3, 1), local_std.transpose(0, 2, 3, 1)],
            axis=-1,
        ).reshape(-1, frames.shape[1] * 3)

    def _make_model(self) -> Any:
        if self.model_type == "sklearn":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            model_class = RandomForestClassifier if self.task == "classification" else RandomForestRegressor
            return model_class(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        if self.model_type == "xgboost":
            module = __import__("xgboost")
            model_class = module.XGBClassifier if self.task == "classification" else module.XGBRegressor
            return model_class(n_estimators=100, max_depth=6, random_state=self.random_state, n_jobs=2)
        module = __import__("lightgbm")
        model_class = module.LGBMClassifier if self.task == "classification" else module.LGBMRegressor
        return model_class(n_estimators=100, random_state=self.random_state, verbosity=-1)

    def fit(self, frames: np.ndarray, target: np.ndarray) -> "TabularBaselineRunner":
        self.model = self._make_model()
        features = self.extract_features(frames)
        labels = np.asarray(target).reshape(-1)
        if len(features) != len(labels):
            raise ValueError("frames and target must contain the same number of pixels")
        self.model.fit(features, labels)
        return self

    def predict(self, frames: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit before predict")
        return self.model.predict(self.extract_features(frames))

    def evaluate(self, frames: np.ndarray, target: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
        actual = np.asarray(target).reshape(-1)
        predicted = self.predict(frames)
        if self.task == "classification":
            return {
                "accuracy": float(accuracy_score(actual, predicted)),
                "f1": float(f1_score(actual, predicted, zero_division=0)),
            }
        return {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        }


class Conv2DUnet(nn.Module):
    """Small U-Net baseline operating on one current frame."""

    def __init__(self, in_channels: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(base_channels * 4, base_channels * 8)
        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec2 = self._block(base_channels * 8, base_channels * 4)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec1 = self._block(base_channels * 4, base_channels * 2)
        self.up0 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec0 = self._block(base_channels * 2, base_channels)
        self.classification_head = nn.Conv2d(base_channels, 1, 1)
        self.qpe_head = nn.Conv2d(base_channels, 1, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        if inputs.ndim != 4:
            raise ValueError("Conv2DUnet expects [batch, channels, height, width]")
        skip0 = self.enc1(inputs)
        skip1 = self.enc2(self.pool(skip0))
        skip2 = self.enc3(self.pool(skip1))
        outputs = self.bottleneck(self.pool(skip2))
        outputs = self.dec2(torch.cat([self.up2(outputs), skip2], dim=1))
        outputs = self.dec1(torch.cat([self.up1(outputs), skip1], dim=1))
        outputs = self.dec0(torch.cat([self.up0(outputs), skip0], dim=1))
        return {"classification": torch.sigmoid(self.classification_head(outputs)), "qpe": F.softplus(self.qpe_head(outputs))}


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, 3, padding=1)

    def forward(self, inputs: Tensor, state: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        if state is None:
            state = (torch.zeros(inputs.size(0), self.hidden_channels, inputs.size(2), inputs.size(3), device=inputs.device),
                     torch.zeros(inputs.size(0), self.hidden_channels, inputs.size(2), inputs.size(3), device=inputs.device))
        hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = self.gates(torch.cat([inputs, hidden], dim=1)).chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        return torch.sigmoid(output_gate) * torch.tanh(cell), cell


class StandardConvLSTM(nn.Module):
    """Four-layer ConvLSTM baseline without temporal attention."""

    def __init__(self, in_channels: int = 6, hidden_channels: tuple[int, ...] = (32, 64, 64, 64)) -> None:
        super().__init__()
        self.cells = nn.ModuleList([ConvLSTMCell(in_channels if index == 0 else hidden_channels[index - 1], width) for index, width in enumerate(hidden_channels)])
        self.classification_head = nn.Conv2d(hidden_channels[-1], 1, 1)
        self.qpe_head = nn.Conv2d(hidden_channels[-1], 1, 1)

    def forward(self, sequence: Tensor) -> dict[str, Tensor]:
        if sequence.ndim != 5:
            raise ValueError("StandardConvLSTM expects [batch, time, channels, height, width]")
        states = [None] * len(self.cells)
        for timestep in sequence.unbind(dim=1):
            outputs = timestep
            for index, cell in enumerate(self.cells):
                hidden, cell_state = cell(outputs, states[index])
                states[index] = (hidden, cell_state)
                outputs = hidden
        return {"classification": torch.sigmoid(self.classification_head(outputs)), "qpe": F.softplus(self.qpe_head(outputs))}
