# LLM Vision — Multimodal Architecture from Scratch

An architecture study extending a GPT-3 style language model with vision capabilities, built entirely from scratch in PyTorch. Implements the core design of GPT-4V / LLaVA — a ViT image encoder fused into a causal language decoder via cross-attention at every layer.

> **Companion to [mini-llm](https://github.com/prajwal9680/mini-llm)** — that repo builds the language model; this one gives it eyes.

---

## What's Implemented

### 1. Vision Encoder (`vision/`)

```
VisionTransformer
├── PatchEmbedding         Conv2d patch projection (16×16 patches)
├── N × ViTBlock
│   ├── LayerNorm
│   ├── VisionAttention    Bidirectional multi-head attention (no causal mask)
│   ├── LayerNorm
│   └── MLP (GELU, 4× expansion)
└── LayerNorm
```

### 2. Multimodal Fusion (`multimodal/`)

```
GPT4Vision
├── Token Embedding
├── VisionTransformer + VisionAdapter   (768 → 4096 projection)
├── 24 × MultiModalBlock
│   ├── CausalSelfAttention   text attends to previous text
│   ├── CrossAttention        text (Q) attends to image patches (K, V)
│   └── MLP
└── LM Head
```

### 3. CLIP Alignment (`alignment/`)
- Projects image and text into a shared 512d embedding space
- L2 normalization + scaled cosine similarity
- Symmetric contrastive loss (same as original CLIP)

### 4. Agentic RAG (`agent/`)
- `VectorStore` — in-memory embedding store with cosine similarity search
- `Retriever` — wraps VectorStore for document lookup
- `Tool` / `ToolRegistry` — named callable tools the agent can dispatch
- `AgentMemory` — tracks reasoning history across steps
- `Agent` — **ReAct loop**: generate → parse ACTION → run tool → observe → repeat

### 5. LangChain Example (`examples/`)
- ChromaDB vector store + OpenAI embeddings
- RetrievalQA chain
- Zero-shot ReAct agent with custom retrieval tool

---

## Project Structure

```
llm-vision-from-scratch/
├── vision/
│   ├── patch_embedding.py      PatchEmbedding (strided Conv2d)
│   ├── attention.py            VisionAttention (bidirectional)
│   ├── vit_block.py            ViTBlock
│   └── vision_transformer.py  VisionTransformer
├── multimodal/
│   ├── adapter.py              VisionAdapter (768 → 4096)
│   ├── cross_attention.py      CrossAttention + CausalSelfAttention
│   ├── block.py                MultiModalBlock + MLP
│   └── model.py                GPT4Vision (full model)
├── alignment/
│   └── clip.py                 CLIPAlignment + contrastive loss
├── agent/
│   ├── retriever.py            VectorStore + Retriever
│   ├── tools.py                Tool + ToolRegistry
│   ├── memory.py               AgentMemory
│   └── agent.py                Agent (ReAct loop)
├── examples/
│   └── langchain_rag.py        LangChain RAG + agent demo
├── requirements.txt
└── README.md
```

---

## Key Design Choices

| Component | Design | Why |
|---|---|---|
| **Bidirectional Attention** (ViT) | No causal mask | Images don't have a sequence direction |
| **Causal Attention** (LLM) | Lower-triangular mask | Language generation is left-to-right |
| **Cross-Attention** | Q from text, K/V from image | Text "queries" the image for relevant patches |
| **VisionAdapter** | Single linear layer | Cheaply bridges two different embedding spaces |
| **CLIP Loss** | Symmetric cross-entropy | Pushes matching pairs together, non-matching apart |
| **ReAct Loop** | Think → Act → Observe | Allows the model to use tools to answer questions |

---

## Quickstart

```bash
pip install torch
python -c "from multimodal.model import GPT4Vision; print('Import OK')"
```

For the LangChain agent example:
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
python examples/langchain_rag.py
```

---

## What I Learned

- How ViT processes images as sequences of patch tokens
- Why cross-attention is the right fusion mechanism (text queries image)
- The difference between bidirectional (encoder) and causal (decoder) attention
- How CLIP aligns image and text representations with contrastive learning
- The ReAct agent pattern: interleaving reasoning and tool use
- How LangChain's RAG chain and agent abstraction work
