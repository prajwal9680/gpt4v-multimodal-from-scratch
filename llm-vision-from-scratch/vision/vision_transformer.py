import torch.nn as nn
from vision.patch_embedding import PatchEmbedding
from vision.vit_block import ViTBlock


class VisionTransformer(nn.Module):
    """
    A ViT-style vision encoder that processes an image into a sequence
    of patch token embeddings.

    Pipeline:
        image (B, 3, H, W)
            → PatchEmbedding  → (B, num_patches, embed_dim)
            → N × ViTBlock    → (B, num_patches, embed_dim)
            → LayerNorm       → (B, num_patches, embed_dim)

    Args:
        layers:    Number of transformer blocks.
        dim:       Embedding dimension.
    """
    def __init__(self, layers=6, dim=768):
        super().__init__()
        self.patch = PatchEmbedding()
        self.blocks = nn.ModuleList([ViTBlock(dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, image):
        x = self.patch(image)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
