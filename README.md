# GPT-4V Style Multimodal Architecture from Scratch

A from-scratch PyTorch implementation of a GPT-4 Vision style multimodal language model, plus CLIP-style vision-language alignment and a ReAct-style agentic framework with RAG.

Built as an architecture study while learning about modern LLMs at RVCE.

---

## What's Implemented

### 1. `multimodalAdditions.py` — Core Architecture

#### Vision Encoder (ViT-style)
```
VisionTransformer
├── PatchEmbedding      (Conv2d patch projection, 16×16 patches)
├── 6× ViTBlock
│   ├── LayerNorm
│   ├── VisionAttention (bidirectional, 12 heads)
│   ├── LayerNorm
│   └── MLP (GELU, 4× expansion)
└── LayerNorm
```

#### Multimodal LLM (GPT-4V style)
```
GPT4Vision
├── Token Embedding
├── VisionTransformer           (image → patch tokens)
├── VisionAdapter               (project 768 → 4096 dim)
├── 24× MultiModalBlock
│   ├── Causal Self-Attention   (text attends to text)
│   ├── Cross-Attention         (text queries image tokens)
│   └── MLP (GELU)
└── LM Head                     (logits over vocab)
```

#### CLIP-style Alignment
```
CLIPAlignment
├── Image projection  (768 → 512)
├── Text projection   (4096 → 512)
├── L2 Normalization
└── Contrastive loss  (symmetric cross-entropy)
```

#### Agentic RAG Framework
```
vectorStore       — cosine similarity search over document embeddings
retriever         — wraps vector store for document lookup
Tools/ToolRegistry — register and dispatch tools by name
AgentMemory       — maintains reasoning history across steps
Agent             — ReAct loop: think → act → observe → repeat
```

### 2. `langchain_agenticAi.py` — LangChain RAG Agent
A minimal LangChain example: ChromaDB vector store + OpenAI embeddings + RetrievalQA chain + zero-shot ReAct agent.

---

## Architecture Highlights

| Component | Detail |
|---|---|
| **Cross-Attention** | Text tokens (Q) attend to image tokens (K, V) — how LLaVA/GPT-4V works |
| **Causal Masking** | Text self-attention is autoregressive (future tokens masked) |
| **Vision Adapter** | Linear projection bridges ViT (768d) and LLM (4096d) embedding spaces |
| **Contrastive Loss** | Symmetric image-text InfoNCE loss, identical to CLIP |
| **ReAct Agent** | Parse ACTION/ACTION_INPUT from model output, dispatch to tools, loop until FINAL ANSWER |

---

## Project Structure

```
gpt-4-multimodal/
├── multimodalAdditions.py   # Full multimodal architecture
├── langchain_agenticAi.py   # LangChain RAG + agent example
└── README.md
```

---

## Quickstart

```bash
pip install torch

# For the LangChain example, also:
pip install langchain openai chromadb
export OPENAI_API_KEY="your-key-here"

python langchain_agenticAi.py
```

---

## Concepts Demonstrated

- **Patch Embedding**: splitting images into tokens with a strided Conv2d
- **Bidirectional vs Causal Attention**: ViT uses full attention, GPT uses masked
- **Cross-Attention**: how multimodal models fuse image and text representations
- **CLIP contrastive loss**: aligning image and text into a shared embedding space
- **LoRA intuition**: low-rank adapter design pattern
- **ReAct Agent loop**: model → parse action → run tool → feed observation → repeat
- **Vector store RAG**: embedding documents, cosine search, retrieval-augmented generation

---

## Notes

> This is an **architecture reference implementation** — a study of how GPT-4V, LLaVA, and similar models are designed. The model weights are not trained; this is meant to demonstrate the structural design and forward pass logic.
