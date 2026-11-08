## Idea

**Multi-head Latent Attention (MLA)** compresses KV into low-rank **latents** plus a small RoPE stream, instead of caching full per-head K/V. After GQA, long-context cache can still explode; DeepSeek-V2/V3 push MLA here while Llama/Qwen dense mainlines stay on GQA. Read with MoE: capacity vs KV.

This lesson is a **shape primer**, not a bit-exact DeepSeek port. Pick a latent rank, project, cache latent + RoPE key, expand when attending.

### Cache shape

[`MLACache` / `MultiHeadLatentAttention`](https://github.com/chaxus/arche/blob/main/arche/mla.py):

```python
@dataclass
class MLACache:
    kv_latent: torch.Tensor  # (batch, seq, kv_lora_rank)
    k_rope: torch.Tensor     # (batch, seq, rope_dim)
```

When rank ≪ heads×dim, this is much smaller than full `(batch, kv_heads, seq, head_dim)` KV.

### Versus GQA

| | GQA (Llama path) | MLA (DeepSeek path) |
| --- | --- | --- |
| Store | Shared KV heads | Latent + tiny RoPE |
| Goal | Cut heads | Cut rank further |
| Swap cost | Architecture-level either way |

### From zero

| Step | Do |
| --- | --- |
| 1 | Choose `kv_lora_rank`, optional `q_lora_rank` |
| 2 | Decouple RoPE dims from content latent |
| 3 | Cache latents across decode steps |
| 4 | Compare memory vs GQA at the same quality bar |

### Check yourself

- Shapes stay consistent; cache appends along seq.
- Cache is smaller; quality near full KV for your product bar.
- Don’t drop MLA under Llama GQA weights.

### Levers

| Lever | Impact |
| --- | --- |
| Smaller latent | Save memory / risk quality |
| + paging | Same serve tricks as GQA stacks |
| Mix under wrong weights | Silent wrong attention |

### Industry map

| Line | Attention / KV |
| --- | --- |
| **DeepSeek-V2 / V3** | MLA (+ MoE body) |
| **Llama 3 / Qwen dense** | GQA mainline |

### This course

```bash
python chapters/01-inference/ep15_mla.py
```

## Run

```bash
python chapters/01-inference/ep15_mla.py
```

## Next

**Frontier checkpoints** (`/frontier-ckpt/`).
