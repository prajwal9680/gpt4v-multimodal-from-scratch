import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Cross-attention layer where text tokens (Q) attend to image tokens (K, V).

    This is the core of the multimodal fusion — text can "look at" the image
    representation and pull in relevant visual information at each layer.

    Args:
        dim:   Embedding dimension (must match LLM dim).
        heads: Number of attention heads.
    """
    def __init__(self, dim=4096, heads=32):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, text, image):
        # text:  (B, T, dim)  — language tokens
        # image: (B, I, dim)  — image patch tokens (after adapter)
        B, T, C = text.shape
        _, I, _ = image.shape

        q = self.q(text).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(image).view(B, I, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(image).view(B, I, self.heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class CausalSelfAttention(nn.Module):
    """
    Standard autoregressive (causal) self-attention for the language decoder.
    Each text token can only attend to previous tokens (causal mask).

    Args:
        dim:        Embedding dimension.
        heads:      Number of attention heads.
        block_size: Maximum sequence length.
    """
    def __init__(self, dim=4096, heads=32, block_size=2048):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Pre-computed causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
