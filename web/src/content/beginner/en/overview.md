## Idea

Hand-write a **Llama 3.2–style** dense decoder: RMSNorm, RoPE, SwiGLU, GQA. Target teaching size: **1B**.

```mermaid
flowchart LR
  tok[Tokenizer] --> emb[Embedding]
  emb --> stack[Blocks]
  stack --> head[LM head]
```

## Run

```bash
python chapters/01-inference/ep01_overview.py
```

Prints config sizes (layers, heads, dim)—not chat.

## Next

**Tokenizer** (`?c=tokenizer`).
