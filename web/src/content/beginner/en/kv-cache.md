## Idea

**What:** **KV cache** — after prefill, keep each layer’s Keys/Values and **append** one new step at decode instead of recomputing the whole past.

**Why:** A correct generate loop without cache recomputes attention over growing history every token → unusable at product lengths. Every production decoder caches KV.

**How:** Per-layer `KVCache`; prefill writes full K/V; decode concatenates the new slice; GQA/MLA/paging shrink what you store.

### 1. What — structure

[`KVCache`](https://github.com/chaxus/arche/blob/main/arche/cache.py):

```python
@dataclass
class KVCache:
    k: torch.Tensor  # (batch, kv_heads, seq, head_dim)
    v: torch.Tensor

    def append(self, k, v):
        return KVCache(k=torch.cat([self.k, k], dim=-2), v=torch.cat([self.v, v], dim=-2))
```

Attention ([`CausalSelfAttention`](https://github.com/chaxus/arche/blob/main/arche/attention.py)) RoPE-applies the new Q/K, merges into cache, then SDPA over **full** cached K/V.

### 2. Why — cost shape

| Mode | Work per new token |
| --- | --- |
| No cache | Re-embed + full stack over all past ids |
| With cache | One new position’s Q/K/V + attend to cache |
| Memory | Grows with `layers × n_kv × seq × dim` |

GQA already cut `n_kv`; paging (later) fights fragmentation across users.

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | One cache slot per layer |
| 2 | Prefill: `cache=None`, write full prompt K/V |
| 3 | Decode: pass cache + `start_pos`, append one step |
| 4 | Never mix layers’ caches |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Matches full recompute numerically (toy check) |
| **Hard** | No layer mixups; seq dim only grows |
| **Soft** | Memory fits target context × batch |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| GQA / MLA / KV-quant | Smaller cache |
| Paging | Multi-user packing |
| No cache | Toy lengths only |

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
