## Idea

### 1. Why cache

Naive decode recomputes K/V for the whole prompt every step.

### 2. What KV cache does

**Prefill** once → store per-layer K/V. Each **decode** step appends one new K/V and attends over the cache. See the strip on this page.

### 3. Related ideas

| Idea | Role |
| --- | --- |
| Contiguous cache | Simple `cat` along time (our teaching cache) |
| Paged KV | Block table + gather (later primer) |
| GQA/MLA | Shrink what you store |

### 4. What labs ship

Every production decoder uses some KV cache; engines differ on paging/layout.

### 5. What “good” means

- Same answers as full recompute (up to numerics)  
- Work grows with new tokens, not full prompt length each step  

### 6. What we do

`KVCache` in `arche/cache.py`.

## Run

```bash
python chapters/01-inference/ep12_kv_cache.py
```

## Next

**RoPE scaling** (`?c=rope-scaling`).
