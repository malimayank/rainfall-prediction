import torch
import torch.nn as nn

class TemporalCrossAttention(nn.Module):
    """Cross-Attention over K=4 temporal frames."""
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
