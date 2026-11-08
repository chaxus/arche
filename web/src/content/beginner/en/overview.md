## Idea

### 1. What it is for

This chapter picks the specimen: a **Llama 3.2-style** next-token stack—tokenize → embed → block stack → LM head—often called a **dense decoder**. Every later chapter hangs on this trunk.

### 2. Why the course needs it

You need one small, open, production-like model to hand-write end to end. Without a shared specimen, RMSNorm / RoPE / GQA won’t share one config.

### 3. Where it comes from · how we chose

| Factor | Llama 3.2 1B |
| --- | --- |
| Size | ~1B, laptop-reachable |
| Open | Weights available with license |
| Parts | RMSNorm, RoPE, SwiGLU, GQA—like many production dense stacks |
| Path | Trunk → inference → tiny train → frontier primers |

### 4. Standards · what “good choice” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Config prints and is consistent | Vocab, depth, dim match docs |
| **Soft** | Carries the whole course | Later chapters plug in |
| **Soft** | Honest scope | Dense trunk + primers, not full vLLM / full pretrain |

### 5. What we do

Lab prints config sizes—not chat yet.

## Run

```bash
python chapters/01-inference/ep01_overview.py
```

## Next

**Tokenizer** (`?c=tokenizer`).
