## Idea

### 1. What it is for

**MLA-style** attention caches a **low-rank latent** (often plus a small RoPE stream) instead of full per-head K/V, expanding when needed—less decode memory.

### 2. Why the LLM pipeline needs it

GQA already cuts KV heads; long context can still dominate VRAM. DeepSeek-like paths compress what you cache, with more implementation detail.

### 3. Where it comes from · how you build it

| Idea | Role |
| --- | --- |
| GQA | Fewer KV heads |
| **MLA** | Latent KV |
| Paged KV | How memory is chunked |

Teaching shapes/API here—**not** a bit-exact DeepSeek port.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Latent cache smaller than full KV | Compression real |
| **Hard** | Forward runs; shapes consistent | Primer bar |
| **Soft** | Quality near full attention | Product claim; not this lesson |

### 5. What labs ship · what we do

Primer in `arche/mla.py`.

## Run

```bash
python chapters/01-inference/ep15_mla.py
```

## Next

**Device · INT8 · chat** (`?c=device-quant-chat`).
