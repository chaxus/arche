## Idea

**What:** **Grouped-Query Attention (GQA)** — more query heads than KV heads; groups of Q heads **share** one KV head. Still causal.

**Why:** Full MHA stores a KV pair per Q head → decode VRAM explodes with length. GQA cuts cache size with small quality hit. Llama 3.2 ~1B uses **32 Q / 8 KV**.

**How:** Choose divisible `n_q` / `n_kv`, project, RoPE, **repeat_interleave** KV to match Q, then SDPA; cache stores only `n_kv`.

### 1. What — shapes in this repo

[`CausalSelfAttention`](https://github.com/chaxus/arche/blob/main/arche/attention.py):

```python
q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
# ... RoPE ...
k_full = new_cache.k.repeat_interleave(self.n_rep, dim=1)
v_full = new_cache.v.repeat_interleave(self.n_rep, dim=1)
attn = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=...)
```

`n_rep = num_heads // num_kv_heads`. Cache appends **narrow** K/V; broadcast only for the matmul.

### 2. Why — MHA / GQA / MQA / MLA

| Layout | Idea | Cost |
| --- | --- | --- |
| **MHA** | `n_kv = n_q` | Richest, biggest cache |
| **GQA** | `n_kv` divides `n_q` | Open midsize default |
| **MQA** | `n_kv = 1` | Extreme cache save |
| **MLA** | Compressed latent KV | Frontier (DeepSeek)—later part |

Head-count change ⇒ **retrain** (or lose the checkpoint).

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Fix `embed_dim = n_q × head_dim` |
| 2 | Pick `n_kv` that divides `n_q` |
| 3 | Bias-free Q/K/V/O linears (Llama contract) |
| 4 | Causal mask on prefill; with cache, attend all past keys |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Valid groups; causal; weight head counts match config |
| **Soft** | Cache bytes ≈ `n_kv/n_q` of MHA |
| **Soft** | Quality holds vs MHA at same width |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Fewer KV heads | Decode memory / speed |
| Too few | Quality cliff |
| MLA path | Different compression story (Frontier) |

### 6. This course

```bash
python chapters/01-inference/ep06_gqa.py
```

## Run

```bash
python chapters/01-inference/ep06_gqa.py
```

## Next

**SwiGLU** (`/swiglu/`): the dense FFN.
