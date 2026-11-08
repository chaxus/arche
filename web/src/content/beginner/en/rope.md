## Idea

**RoPE** (Rotary Position Embedding) rotates query and key vectors by token position so **relative distance** shows up in the attention dot product. Embeddings say *which* token; attention still needs *where*. Absolute position tables on the embedding are an older pattern; modern dense Llama, Qwen, and DeepSeek apply RoPE on Q and K. Pick `rope_theta`, cache cos/sin, combine with `rotate_half` on Q/K; long-context scaling is a later knob.

### Apply step in code

Repo ([`arche/rope.py`](https://github.com/chaxus/arche/blob/main/arche/rope.py)):

```python
def apply_rotary_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
```

`build_rope_cache` precomputes cos/sin for `0 .. seq_len-1`. Attention slices by `start_pos` when decoding with cache.

### Design choices

| Choice | Rationale |
| --- | --- |
| Rotate Q/K only | V stays content; positions live in scores |
| Relative via rotation | Dot(q_m, k_n) depends on `m-n` |
| Large `theta` (Llama 3: 500k) | Supports longer contexts than old 10k |
| Match checkpoint `theta`/scale | Wrong PE ⇒ silent quality death under loaded weights |

### Wiring from scratch

| Step | Do |
| --- | --- |
| 1 | Even `head_dim`; build inv-freq from `theta` |
| 2 | Cache cos/sin up to max seq (memory tradeoff) |
| 3 | Apply after Q/K project, before SDPA |
| 4 | Long context: Llama-3 wavelength smoothing (`rope-scaling` later) |

### Check yourself

- Shapes unchanged; even head dim; RoPE on both Q and K.
- Checkpoint `theta` and scale factors match.
- Nearby tokens score differently than distant ones.

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| Theta / scaling | Long-context behavior |
| Absolute PE → RoPE | Modern default; don’t mix families with foreign weights |
| Skip RoPE | Model can’t reliably use order |

### This course

```bash
python chapters/01-inference/ep05_rope.py
```

## Run

```bash
python chapters/01-inference/ep05_rope.py
```

## Next

**GQA** (`/gqa/`): head layout that saves KV cache.
