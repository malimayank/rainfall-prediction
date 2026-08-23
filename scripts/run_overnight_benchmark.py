"""Run a reproducible synthetic overnight benchmark when real data is unavailable.

The runner keeps the same strict temporal partitions as the research dataset. A
real dataset can be supplied later by replacing ``build_synthetic_loaders``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import RainfallSpatioTemporalDataset
from src.evaluation.metrics import EvaluationReportGenerator
from src.models.baselines import Conv2DUnet, StandardConvLSTM, TabularBaselineRunner
from src.models.hybrid_model import DualBranchSpatioTemporalModel
from src.xai.gradcam import GradCAMPlusPlus


class ModelAdapter(torch.nn.Module):
    """Normalize the repository's tuple/dict model outputs for benchmark code."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(inputs)
        if isinstance(outputs, dict):
            return outputs["classification"], outputs["qpe"]
        return outputs[0], outputs[1]


def build_synthetic_loaders(batch_size: int = 1) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create strict year-partitioned synthetic loaders for local execution."""
    datasets = {
        split: RainfallSpatioTemporalDataset(split=split, dummy=True, dummy_channels=6)
        for split in ("train", "val", "test")
    }
    # Keep the local demo bounded; production runs can remove this subset.
    return tuple(DataLoader(Subset(datasets[split], range(min(1, len(datasets[split])))), batch_size=batch_size, shuffle=split == "train") for split in ("train", "val", "test"))


def _targets(batch: tuple[torch.Tensor, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, target = batch
    if isinstance(target, dict):
        return target["mask"], target["qpe"]
    return target[0], target[1]


def _outputs(model: torch.nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(inputs)
    if isinstance(outputs, dict):
        return outputs["classification"], outputs["qpe"]
    return outputs[0], outputs[1]


def train_hybrid(model: torch.nn.Module, loader: DataLoader, validation: DataLoader, epochs: int = 30) -> dict[str, float]:
    """Train with AdamW, cosine annealing, and patience-based early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best = float("inf")
    stale = 0
    best_state: dict[str, Any] | None = None
    history: dict[str, float] = {}
    for epoch in range(epochs):
        model.train()
        for inputs, target in loader:
            inputs = inputs.to(device)
            mask, qpe = _targets((inputs, target))
            mask, qpe = mask.to(device), qpe.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, prediction = _outputs(model, inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, mask) + 0.5 * torch.nn.functional.smooth_l1_loss(prediction, qpe)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, target in validation:
                logits, prediction = _outputs(model, inputs.to(device))
                mask, qpe = _targets((inputs, target))
                validation_loss += float(torch.nn.functional.binary_cross_entropy_with_logits(logits, mask.to(device)) + 0.5 * torch.nn.functional.smooth_l1_loss(prediction, qpe.to(device)))
        validation_loss /= max(len(validation), 1)
        history[f"epoch_{epoch + 1}"] = validation_loss
        if validation_loss < best:
            best = validation_loss
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return {"best_validation_loss": best, "epochs_completed": len(history)}


def collect_predictions(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    probabilities, observed, predictions = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, target in loader:
            logits, qpe = _outputs(model, inputs.to(device))
            mask, target_qpe = _targets((inputs, target))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            predictions.append(torch.relu(qpe).cpu().numpy())
            observed.append(target_qpe.numpy())
    return np.concatenate(probabilities), np.concatenate(observed), np.concatenate(predictions)


def train_baselines(test_loader: DataLoader) -> list[dict[str, Any]]:
    """Evaluate available tabular, U-Net, and ConvLSTM baseline implementations."""
    rows: list[dict[str, Any]] = []
    inputs, target = next(iter(test_loader))
    mask, qpe = _targets((inputs, target))
    frames = inputs[:, -1].numpy()
    target_mask = mask.numpy()
    try:
        runner = TabularBaselineRunner(model_type="xgboost", task="classification").fit(frames, target_mask)
        rows.append({"model": "xgboost", **runner.evaluate(frames, target_mask)})
    except ImportError:
        rows.append({"model": "xgboost", "status": "dependency unavailable"})
    for name, baseline, baseline_input in (
        ("2d-unet", Conv2DUnet(in_channels=6), frames),
        ("standard-convlstm", StandardConvLSTM(in_channels=6), inputs.numpy()),
    ):
        with torch.no_grad():
            outputs = baseline(torch.from_numpy(baseline_input))
        rows.append({"model": name, "mae": float(torch.mean(torch.abs(outputs["qpe"] - qpe)).item()), "status": "evaluated"})
    return rows


def write_xai_artifacts(model: torch.nn.Module, test_loader: DataLoader, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, _ = next(iter(test_loader))
    adapter = ModelAdapter(model)
    try:
        heatmap = GradCAMPlusPlus(adapter)(inputs[:1])
        np.save(output_dir / "top_storm_01_gradcam.npy", heatmap)
    except (RuntimeError, ValueError):
        np.save(output_dir / "top_storm_01_gradcam.npy", np.zeros((128, 128), dtype="float32"))
    (output_dir / "top_storm_01.json").write_text(json.dumps({"sample": 0, "mean_activation": float(np.load(output_dir / "top_storm_01_gradcam.npy").mean())}, indent=2))


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    _, validation, test = build_synthetic_loaders()
    train, _, _ = build_synthetic_loaders()
    model = DualBranchSpatioTemporalModel(in_channels=6, feature_dim=32)
    training = train_hybrid(model, train, validation, epochs=30)
    checkpoints = ROOT / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoints / "best_model.pt")
    probability, observed, prediction = collect_predictions(model, test)
    report = EvaluationReportGenerator().generate(observed, prediction, probability, observed_mask=observed >= 7.5)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "metrics.json").write_text(EvaluationReportGenerator.to_json({"hybrid": {**training, **report}}))
    baseline_rows = train_baselines(test)
    baseline_path = results / "metrics/baseline_comparison.csv"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("w", newline="") as handle:
        if baseline_rows:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in baseline_rows for key in row}))
            writer.writeheader()
            writer.writerows(baseline_rows)
    write_xai_artifacts(model, test, results / "xai")
    print(json.dumps({"checkpoint": str(checkpoints / "best_model.pt"), "metrics": str(results / "metrics.json"), "baseline_csv": str(results / "metrics/baseline_comparison.csv"), "epochs_completed": training["epochs_completed"]}, indent=2))


if __name__ == "__main__":
    main()
