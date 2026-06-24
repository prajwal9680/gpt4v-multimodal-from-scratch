import torch.nn as nn


class VisionAdapter(nn.Module):
    """
    Projects vision embeddings from the ViT embedding space into the
    LLM embedding space via a single linear layer.

    This is the "connector" between the vision encoder and the language model —
    the same approach used in LLaVA and GPT-4V.

    Args:
        vision_dim: Output dimension of the VisionTransformer (default: 768).
        llm_dim:    Embedding dimension of the language model (default: 4096).
    """
    def __init__(self, vision_dim=768, llm_dim=4096):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim)

    def forward(self, x):
        return self.proj(x)
