import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPAlignment(nn.Module):
    """
    CLIP-style contrastive vision-language alignment.

    Projects image and text embeddings into a shared latent space,
    then computes a symmetric contrastive loss to align matching
    image-text pairs and push apart non-matching ones.

    This is identical in design to the original CLIP (Radford et al., 2021).

    Args:
        vision_dim: Dimension of image embeddings (from VisionTransformer).
        text_dim:   Dimension of text embeddings (from the LLM).
        proj_dim:   Shared projection space dimension (default: 512).
    """
    def __init__(self, vision_dim=768, text_dim=4096, proj_dim=512):
        super().__init__()
        self.image_proj = nn.Linear(vision_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, image_emb, text_emb):
        # Project both into shared space
        image_emb = F.normalize(self.image_proj(image_emb), dim=-1)
        text_emb = F.normalize(self.text_proj(text_emb), dim=-1)

        # Scaled cosine similarity matrix
        sim = (image_emb @ text_emb.T) / self.temperature
        return sim

    def clip_loss(self, sim):
        """
        Symmetric cross-entropy loss over the similarity matrix.
        Diagonal entries are the correct image-text matches.
        """
        labels = torch.arange(sim.size(0), device=sim.device)
        loss_i = F.cross_entropy(sim, labels)      # image → text
        loss_t = F.cross_entropy(sim.T, labels)    # text → image
        return (loss_i + loss_t) / 2
