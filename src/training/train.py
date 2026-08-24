"""Train the rainfall model on real INSAT/GPM observations only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import RainfallSpatioTemporalDataset, discover_real_samples
from src.evaluation.metrics import contingency_metrics, brier_score
from src.models.hybrid_model import DualBranchSpatioTemporalModel
from src.training.losses import HybridCompoundLoss


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for inputs, target_mask, target_qpe in loader:
        inputs = inputs.to(device); target_mask = target_mask.to(device); target_qpe = target_qpe.to(device)
        optimizer.zero_grad(set_to_none=True)
        cls_logits, qpe, _ = model(inputs)
        loss, _, _ = loss_fn(cls_logits, qpe, target_mask, target_qpe)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total += float(loss.detach())
    return total / max(len(loader), 1)


def evaluate(model, loader, device) -> dict[str, float]:
    model.eval(); probabilities = []; observed_masks = []
    with torch.no_grad():
        for inputs, target_mask, _ in loader:
            cls_logits, _, _ = model(inputs.to(device))
            probabilities.append(torch.sigmoid(cls_logits).cpu())
            observed_masks.append(target_mask)
    probability = torch.cat(probabilities).numpy().reshape(-1)
    observed = torch.cat(observed_masks).numpy().reshape(-1)
    metrics = contingency_metrics(observed >= 0.5, probability >= 0.5)
    metrics["brier_score"] = brier_score(probability, observed)
    return {key: float(value) for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    satellite_paths, gpm_paths = discover_real_samples(args.data_dir)
    dataset = RainfallSpatioTemporalDataset(satellite_paths, gpm_paths)
    if len(dataset) < 2:
        raise RuntimeError("At least two validated real samples are required for a train/test partition")
    train_count = max(1, int(len(dataset) * 0.8))
    test_count = len(dataset) - train_count
    if test_count == 0:
        train_count -= 1; test_count = 1
    train_set, test_set = random_split(dataset, [train_count, test_count], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchSpatioTemporalModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = HybridCompoundLoss()
    best_loss = float("inf")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device); scheduler.step()
        if loss < best_loss:
            best_loss = loss
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print(f"epoch={epoch + 1}/{args.epochs} loss={loss:.6f}")
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/metrics/test_metrics.json").write_text(json.dumps(evaluate(model, test_loader, device), indent=2))


if __name__ == "__main__":
    main()
