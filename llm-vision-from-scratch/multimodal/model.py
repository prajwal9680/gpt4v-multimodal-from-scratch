import torch.nn as nn
from vision.vision_transformer import VisionTransformer
from multimodal.adapter import VisionAdapter
from multimodal.block import MultiModalBlock


class GPT4Vision(nn.Module):
    """
    GPT-4V style multimodal language model.

    Combines a Vision Transformer encoder with a causal language decoder
    via cross-attention at every layer — the same architecture used in
    GPT-4V, Flamingo, and similar vision-language models.

    Pipeline:
        image  → VisionTransformer → patch tokens (B, num_patches, 768)
                                   → VisionAdapter → (B, num_patches, dim)
        tokens → Token Embedding   → (B, T, dim)
        For each MultiModalBlock:
            text = CausalSelfAttn(text) + CrossAttn(text, image) + MLP(text)
        text → LayerNorm → LM Head → logits (B, T, vocab_size)

    Args:
        vocab_size: Size of the token vocabulary.
        dim:        LLM embedding dimension (default: 4096).
        layers:     Number of MultiModalBlocks (default: 24).
    """
    def __init__(self, vocab_size, dim=4096, layers=24):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.vision = VisionTransformer()
        self.adapter = VisionAdapter(vision_dim=768, llm_dim=dim)
        self.blocks = nn.ModuleList([MultiModalBlock(dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, image, tokens):
        # Encode image → patch tokens
        image_tokens = self.adapter(self.vision(image))  # (B, num_patches, dim)

        # Embed text tokens
        text = self.token_embed(tokens)                  # (B, T, dim)

        # Process through multimodal blocks
        for block in self.blocks:
            text = block(text, image_tokens)

        text = self.norm(text)
        return self.lm_head(text)                        # (B, T, vocab_size)
