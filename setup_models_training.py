import os
from pathlib import Path

files = {
    "src/training/losses.py": """import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialFocalLoss(nn.Module):
    \"\"\"Spatial Focal Loss for extreme class imbalance in precipitation.\"\"\"
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class ExtremeWeightedLogCoshLoss(nn.Module):
    \"\"\"Penalizes underestimation on high-intensity rain cells.\"\"\"
    def __init__(self, beta: float = 2.5, threshold: float = 7.5):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        log_cosh = torch.log(torch.cosh(diff + 1e-12))
        weights = 1.0 + self.beta * (target / self.threshold)
        return torch.mean(weights * log_cosh)

class HybridCompoundLoss(nn.Module):
    def __init__(self, lambda_cls: float = 1.0, lambda_reg: float = 0.5):
        super().__init__()
        self.focal = SpatialFocalLoss()
        self.reg = ExtremeWeightedLogCoshLoss()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg

    def forward(self, pred_cls_logits: torch.Tensor, pred_qpe: torch.Tensor, true_cls: torch.Tensor, true_qpe: torch.Tensor):
        l_cls = self.focal(pred_cls_logits, true_cls)
        l_reg = self.reg(pred_qpe, true_qpe)
        total = self.lambda_cls * l_cls + self.lambda_reg * l_reg
        return total, l_cls, l_reg
""",

    "src/models/convnext_encoder.py": """import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x

class ConvNeXtSpatialEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, dims: list = [64, 128, 256]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, dims[0])
        )
        self.stage1 = nn.Sequential(ConvNeXtBlock(dims[0]), ConvNeXtBlock(dims[0]))
        self.down1 = nn.Sequential(nn.GroupNorm(1, dims[0]), nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2))
        self.stage2 = nn.Sequential(ConvNeXtBlock(dims[1]), ConvNeXtBlock(dims[1]))
        self.down2 = nn.Sequential(nn.GroupNorm(1, dims[1]), nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2))
        self.stage3 = nn.Sequential(ConvNeXtBlock(dims[2]), ConvNeXtBlock(dims[2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        return x
""",

    "src/models/temporal_attention.py": """import torch
import torch.nn as nn

class TemporalCrossAttention(nn.Module):
    \"\"\"Cross-Attention over K=4 temporal frames.\"\"\"
    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        # x shape: [B, K=4, Dim, H, W]
        B, K, C, H, W = x.shape
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, K, C)
        x_flat = x_flat + self.pos_embed

        attn_out, attn_weights = self.attn(x_flat, x_flat, x_flat)
        out = self.norm(x_flat + attn_out)
        out = out.reshape(B, H, W, K, C).permute(0, 3, 4, 1, 2)
        # Take latest refined frame (t)
        return out[:, -1], attn_weights
""",

    "src/models/hybrid_model.py": """import torch
import torch.nn as nn
from src.models.convnext_encoder import ConvNeXtSpatialEncoder
from src.models.temporal_attention import TemporalCrossAttention

class DualBranchSpatioTemporalModel(nn.Module):
    def __init__(self, in_channels: int = 6, feature_dim: int = 256):
        super().__init__()
        self.encoder = ConvNeXtSpatialEncoder(in_channels=in_channels, dims=[64, 128, feature_dim])
        self.temporal_attn = TemporalCrossAttention(dim=feature_dim)
        
        # Decoder (Upsampling to 128x128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 128, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.GELU()
        )

        # Dual Heads
        self.cls_head = nn.Conv2d(32, 1, kernel_size=1) # Logits
        self.qpe_head = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.ReLU()) # Non-negative precipitation mm/h

    def forward(self, x: torch.Tensor):
        # x shape: [B, K=4, C=6, H=128, W=128]
        B, K, C, H, W = x.shape
        x_reshaped = x.view(B * K, C, H, W)
        feats = self.encoder(x_reshaped)
        _, fC, fH, fW = feats.shape
        feats_seq = feats.view(B, K, fC, fH, fW)

        temporal_fused, attn_weights = self.temporal_attn(feats_seq)
        dec_out = self.decoder(temporal_fused)

        cls_logits = self.cls_head(dec_out)
        qpe_rate = self.qpe_head(dec_out)

        return cls_logits, qpe_rate, attn_weights
""",

    "src/training/train.py": """import os
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
""",

    "tests/test_models.py": """import torch
from src.models.hybrid_model import DualBranchSpatioTemporalModel
from src.training.losses import HybridCompoundLoss

def test_model_forward_and_loss():
    model = DualBranchSpatioTemporalModel(in_channels=6, feature_dim=256)
    dummy_x = torch.randn(2, 4, 6, 128, 128)
    dummy_y_cls = torch.randint(0, 2, (2, 1, 128, 128)).float()
    dummy_y_qpe = torch.rand(2, 1, 128, 128) * 20.0

    cls_logits, qpe_rate, attn_weights = model(dummy_x)

    assert cls_logits.shape == (2, 1, 128, 128)
    assert qpe_rate.shape == (2, 1, 128, 128)
    assert (qpe_rate >= 0.0).all()

    loss_fn = HybridCompoundLoss()
    total_loss, l_cls, l_reg = loss_fn(cls_logits, qpe_rate, dummy_y_cls, dummy_y_qpe)

    assert not torch.isnan(total_loss)
    total_loss.backward()

    # Verify gradients computed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
"""
}

for path_str, content in files.items():
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(content)
    print(f"Generated: {path_str}")

print("\nModels and Training Suite successfully set up!")
