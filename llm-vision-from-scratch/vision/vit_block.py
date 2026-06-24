import torch.nn as nn
from vision.attention import VisionAttention


class ViTBlock(nn.Module):
    """
    A single Vision Transformer block:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Args:
        dim: Embedding dimension.
    """
    def __init__(self, dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VisionAttention(dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
