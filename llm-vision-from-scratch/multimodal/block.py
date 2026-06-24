import torch.nn as nn
import torch.nn.functional as F
from multimodal.cross_attention import CausalSelfAttention, CrossAttention


class MLP(nn.Module):
    """Feed-forward network with GELU activation (4x expansion)."""
    def __init__(self, dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MultiModalBlock(nn.Module):
    """
    A single multimodal transformer block — the core building block of GPT-4V.

    Processing order:
        1. Causal Self-Attention  — text attends to previous text tokens
        2. Cross-Attention        — text attends to image patch tokens
        3. MLP Feed-Forward       — per-token transformation

    Each sub-layer uses Pre-LayerNorm (LayerNorm before, not after).

    Args:
        dim: Embedding dimension (must match LLM and adapted image dim).
    """
    def __init__(self, dim=4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CausalSelfAttention(dim)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim)

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, text, image):
        text = text + self.self_attn(self.norm1(text))
        text = text + self.cross_attn(self.norm2(text), image)
        text = text + self.mlp(self.norm3(text))
        return text
