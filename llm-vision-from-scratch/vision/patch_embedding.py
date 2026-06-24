import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits an image into fixed-size patches and projects each patch
    into an embedding vector using a strided convolution.

    Args:
        image_size: Input image resolution (assumes square).
        patch_size: Size of each patch (e.g. 16 → 16×16 patches).
        embed_dim:  Embedding dimension for each patch token.
    """
    def __init__(self, image_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)          # (B, embed_dim, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        return x
