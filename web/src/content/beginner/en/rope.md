## Idea

**What:** **RoPE** (Rotary Position Embedding) — rotate Q and K by token position so **relative distance** enters the attention dot product.

**Why:** Embeddings say *which* token; attention still needs *where*. Absolute PE tables on embeds are older; modern dense Llama/Qwen/DeepSeek use RoPE on Q/K.

**How:** Pick `rope_theta`, cache cos/sin, `rotate_half` + combine on Q/K; long-context scaling is a later knob.

### 1. What — apply step

Repo ([`arche/rope.py`](https://github.com/chaxus/arche/blob/main/arche/rope.py)):

```python
def apply_rotary_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
```

`build_rope_cache` precomputes cos/sin for `0 .. seq_len-1`. Attention slices by `start_pos` when decoding with cache.

### 2. Why — design

| Choice | Why |
| --- | --- |
| Rotate Q/K only | V stays content; positions live in scores |
| Relative via rotation | Dot(q_m, k_n) depends on `m-n` |
| Large `theta` (Llama 3: 500k) | Supports longer contexts than old 10k |
| Match checkpoint `theta`/scale | Wrong PE ⇒ silent quality death under loaded weights |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Even `head_dim`; build inv-freq from `theta` |
| 2 | Cache cos/sin up to max seq (memory tradeoff) |
| 3 | Apply after Q/K project, before SDPA |
| 4 | Long context: Llama-3 wavelength smoothing (`rope-scaling` later) |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Shape unchanged; even head dim; applied to Q and K |
| **Hard** | Checkpoint `theta` / scale factors match |
| **Soft** | Nearby tokens score differently than far ones |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Theta / scaling | Long-context behavior |
| Absolute PE → RoPE | Modern default; don’t mix families with foreign weights |
| Skip RoPE | Model can’t reliably use order |

### 6. This course

```bash
python chapters/01-inference/ep05_rope.py
```

## Run

```bash
python chapters/01-inference/ep05_rope.py
```

## Next

**GQA** (`/gqa/`): head layout that saves KV cache.
