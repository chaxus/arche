## Idea

**What:** **Multi-head Latent Attention (MLA)** — compress KV into low-rank **latents** (+ a small RoPE stream) instead of caching full per-head K/V.

**Why:** After GQA, long-context KV can still explode. DeepSeek-V2/V3 push MLA for cache; Llama/Qwen dense mainlines stay on **GQA**. Together with MoE: **MoE = capacity, MLA = KV**.

**How:** Pick latent rank; project to latent; cache latent + RoPE key; expand when attending. This lesson is a **shape primer**, not bit-exact DeepSeek.

### 1. What — cache shape

[`MLACache` / `MultiHeadLatentAttention`](https://github.com/chaxus/arche/blob/main/arche/mla.py):

```python
@dataclass
class MLACache:
    kv_latent: torch.Tensor  # (batch, seq, kv_lora_rank)
    k_rope: torch.Tensor     # (batch, seq, rope_dim)
```

Much smaller than `(batch, kv_heads, seq, head_dim)` full KV when rank ≪ heads×dim.

### 2. Why — vs GQA

| | GQA (Llama path) | MLA (DeepSeek path) |
| --- | --- | --- |
| Store | Shared KV heads | Latent + tiny RoPE |
| Goal | Cut heads | Cut rank further |
| Swap cost | Architecture-level either way |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Choose `kv_lora_rank`, optional `q_lora_rank` |
| 2 | Decouple RoPE dims from content latent |
| 3 | Cache latents across decode steps |
| 4 | Compare memory vs GQA at same quality bar |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Consistent shapes; cache append along seq |
| **Soft** | Smaller cache; quality near full KV for the product bar |
| **Soft** | Engines gaining MLA support over time |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Smaller latent | Save memory / risk quality |
| + paging | Same serve tricks as GQA stacks |
| Don’t mix under Llama GQA weights | Silent wrong attention |

### 6. Industry map

| Line | Attention / KV |
| --- | --- |
| **DeepSeek-V2 / V3** | MLA (+ MoE body) |
| **Llama 3 / Qwen dense** | GQA mainline |

### 7. This course

```bash
python chapters/01-inference/ep15_mla.py
```

## Run

```bash
python chapters/01-inference/ep15_mla.py
```

## Next

**Frontier checkpoints** (`/frontier-ckpt/`).
