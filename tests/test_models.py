import torch
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
