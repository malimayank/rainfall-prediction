import torch
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
