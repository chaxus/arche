## Idea

### 1. Why MLA

GQA shrinks KV heads; decode memory can still dominate.

### 2. What MLA does

Cache a **low-rank latent** (+ small RoPE stream) instead of full per-head K/V; reconstruct what attention needs.

### 3. Landscape

| Idea | Role |
| --- | --- |
| GQA | Fewer KV heads |
| **MLA** | Latent KV (DeepSeek-style idea) |
| Paged KV | How storage is allocated |

### 4. What labs ship

Teaching shapes—not bit-identical DeepSeek weights.

### 5. What “good” means

- Latent cache smaller than full KV  
- Forward still runs  

### 6. What we do

`arche/mla.py` primer.

## Run

```bash
python chapters/01-inference/ep15_mla.py
```

## Next

**Device · INT8 · chat** (`?c=device-quant-chat`).
