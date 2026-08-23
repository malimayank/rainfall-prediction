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

        attn_out, attn_weights = self.attn(x_flat, x_flat, x_flat, need_weights=True)
        out = self.norm(x_flat + attn_out)
        out = out.reshape(B, H, W, K, C).permute(0, 3, 4, 1, 2)
        # Take latest refined frame (t)
        return out[:, -1], attn_weights


class SpatioTemporalCrossAttention(nn.Module):
    """Temporal attention API used by the hybrid model and XAI engine."""

    def __init__(self, embed_dim: int, num_heads: int = 8, max_sequence_length: int = 4):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.position_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, embed_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.last_attention_rollout = None

    def forward(self, sequence: torch.Tensor, return_attention: bool = False):
        if sequence.ndim != 5:
            raise ValueError("sequence must have shape [batch, time, channels, height, width]")
        batch, steps, channels, height, width = sequence.shape
        if steps > self.position_embedding.shape[1]:
            raise ValueError("sequence length exceeds max_sequence_length")
        tokens = sequence.permute(0, 3, 4, 1, 2).reshape(batch * height * width, steps, channels)
        tokens = tokens + self.position_embedding[:, :steps]
        attended, weights = self.attention(tokens, tokens, tokens, need_weights=True, average_attn_weights=False)
        attended = self.norm(attended + tokens)
        output = attended[:, -1].reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        rollout = weights.mean(dim=1)[:, -1].reshape(batch, height, width, steps)
        self.last_attention_rollout = rollout.detach()
        return (output, rollout) if return_attention else output

    def get_attention_rollout(self):
        return self.last_attention_rollout
