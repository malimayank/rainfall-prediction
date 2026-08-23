import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.models.hybrid_model import DualBranchSpatioTemporalModel
from src.training.losses import HybridCompoundLoss

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_cls, total_reg = 0.0, 0.0, 0.0
    for x, y_cls, y_qpe in loader:
        x, y_cls, y_qpe = x.to(device), y_cls.to(device), y_qpe.to(device)
        optimizer.zero_grad()
        pred_cls, pred_qpe, _ = model(x)
        loss, l_cls, l_reg = loss_fn(pred_cls, pred_qpe, y_cls, y_qpe)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_cls += l_cls.item()
        total_reg += l_reg.item()
    return total_loss / len(loader), total_cls / len(loader), total_reg / len(loader)

def save_checkpoint(model, optimizer, epoch, path: str = "checkpoints/best_model.pt"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)
    print(f"Checkpoint saved to {path}")
