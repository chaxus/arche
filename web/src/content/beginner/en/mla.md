## Idea

From-zero long-context serving: after GQA, KV can still explode—**MLA** (Multi-head Latent Attention) compresses cache with latents. Together with **MoE**, this is the usual one-line picture of **DeepSeek-V2 / V3**: MoE for capacity, MLA for KV.

### 1. What it is for

Store low-rank latents (+ small RoPE stream), expand when needed. Llama / Qwen mainlines stay on **GQA**; DeepSeek’s open frontier line pushes **MLA**.

### 2. From zero: how you build it

Pick latent dim; train MLA forward; cache latents; this lesson is a shape primer, not bit-exact.

### 3. What “good” means

Smaller cache; consistent shapes; quality near full KV at product bar.

### 4. Improve · whole-model impact

Smaller latent ⇒ save memory / risk quality; composable with paging; architecture-level change.

### 5. Industry · open models

| Line | Attention / KV |
| --- | --- |
| **DeepSeek-V2 / V3** | **MLA** (+ MoE body) |
| **Llama 3 / Qwen dense** | **GQA** mainline |
| Engines (vLLM, …) | Adding MLA support over time |

### 6. This course

```bash
python chapters/01-inference/ep15_mla.py
```

## Run

```bash
python chapters/01-inference/ep15_mla.py
```

## Next


**Frontier checkpoints** (`/frontier-ckpt/`).
