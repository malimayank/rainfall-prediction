import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialFocalLoss(nn.Module):
    """Spatial Focal Loss for extreme class imbalance in precipitation."""
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
    """Penalizes underestimation on high-intensity rain cells."""
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
