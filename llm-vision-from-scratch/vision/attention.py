import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionAttention(nn.Module):
    """
    Bidirectional (non-causal) multi-head self-attention for the Vision Transformer.
    Unlike the language model, there is no causal mask — every patch can attend to
    every other patch.

    Args:
        dim:   Embedding dimension.
        heads: Number of attention heads.
    """
    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Full (bidirectional) attention — no causal mask
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)
