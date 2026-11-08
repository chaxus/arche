## Idea

From zero to serving: a correct generate loop isn’t enough—**without KV cache long decode is unusable**.

### 1. What it is for

Prefill stores K/V; decode appends one step and attends to the cache.

### 2. From zero: how you build it

Per-layer buffers; prefill write; decode append; later paging + GQA/MLA to shrink.

### 3. What “good” means

Matches full recompute; no layer mixups; cost grows with new tokens; memory OK.

### 4. Improve · whole-model impact

GQA/MLA/paging/KV-quant cut memory; without cache only toy lengths work.

### 5. Industry · open models

Every production decoder caches KV (vLLM, TGI, llama.cpp…).

### 6. This course

```bash
python chapters/01-inference/ep12_kv_cache.py
```

## Run

```bash
python chapters/01-inference/ep12_kv_cache.py
```

## Next

**RoPE scaling** (`/rope-scaling/`).
