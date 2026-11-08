## Idea

**Grouped-Query Attention (GQA)** uses more query heads than KV heads: groups of Q heads **share** one K/V head. It is still fully causal attention. Full multi-head attention stores a separate KV pair per Q head, so decode memory grows fast with sequence length; GQA shrinks the KV cache with a usually small quality cost. Llama 3.2 ~1B runs **32 Q / 8 KV**. Pick divisible `n_q` and `n_kv`, project, apply RoPE, **repeat_interleave** KV to line up with Q, then SDPA; the cache stores only `n_kv` heads.

### Shapes in this repo

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

### MHA, GQA, MQA, and MLA

| Layout | Idea | Cost |
| --- | --- | --- |
| **MHA** | `n_kv = n_q` | Richest, biggest cache |
| **GQA** | `n_kv` divides `n_q` | Open midsize default |
| **MQA** | `n_kv = 1` | Extreme cache save |
| **MLA** | Compressed latent KV | Frontier (DeepSeek)—later part |

Changing head counts ⇒ **retrain** (or lose the checkpoint).

### Wiring from scratch

| Step | Do |
| --- | --- |
| 1 | Fix `embed_dim = n_q × head_dim` |
| 2 | Pick `n_kv` that divides `n_q` |
| 3 | Bias-free Q/K/V/O linears (Llama contract) |
| 4 | Causal mask on prefill; with cache, attend all past keys |

### Check yourself

- Valid groups; causal; weight head counts match config.
- Cache bytes ≈ `n_kv/n_q` of MHA.
- Quality holds vs MHA at the same width.

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| Fewer KV heads | Decode memory / speed |
| Too few | Quality cliff |
| MLA path | Different compression story (Frontier) |

### This course

```bash
python chapters/01-inference/ep06_gqa.py
```

## Run

```bash
python chapters/01-inference/ep06_gqa.py
```

## Next

**SwiGLU** (`/swiglu/`): the dense FFN.
