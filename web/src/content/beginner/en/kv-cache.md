## Idea

**KV cache** keeps each layer’s Keys and Values after prefill, then **appends** one new step at decode instead of recomputing the full past every token. A correct generate loop without cache recomputes attention over growing history on every step—fine for demos, unusable at product lengths. Production decoders all cache KV; GQA, MLA, and paging shrink what you store on top of that.

### Cache structure

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

### Cost shape

| Mode | Work per new token |
| --- | --- |
| No cache | Re-embed + full stack over all past ids |
| With cache | One new position’s Q/K/V + attend to cache |
| Memory | Grows with `layers × n_kv × seq × dim` |

GQA already cut `n_kv`; paging (later) fights fragmentation across users.

### Wire decode

| Step | Do |
| --- | --- |
| 1 | One cache slot per layer |
| 2 | Prefill: `cache=None`, write full prompt K/V |
| 3 | Decode: pass cache + `start_pos`, append one step |
| 4 | Never mix layers’ caches |

### Check yourself

| Must hold | Matches full recompute numerically (toy check) |
| Must hold | No layer mixups; seq dim only grows |
| Worth aiming for | Memory fits target context × batch |

### Tuning memory

| Lever | Impact |
| --- | --- |
| GQA / MLA / KV-quant | Smaller cache |
| Paging | Multi-user packing |
| No cache | Toy lengths only |

### This course

```bash
python chapters/01-inference/ep12_kv_cache.py
```

## Run

```bash
python chapters/01-inference/ep12_kv_cache.py
```

## Next

**RoPE scaling** (`/rope-scaling/`).
