## Idea

From-zero long-context serving: after GQA, KV can still explode—**MLA** compresses cache with latents (DeepSeek line).

### 1. What it is for

Store low-rank latents (+ small RoPE stream), expand when needed.

### 2. From zero: how you build it

Pick latent dim; train MLA forward; cache latents; this lesson is a shape primer, not bit-exact.

### 3. What “good” means

Smaller cache; consistent shapes; quality near full KV at product bar.

### 4. Improve · whole-model impact

Smaller latent ⇒ save memory / risk quality; composable with paging; architecture-level change.

### 5. Industry · open models

DeepSeek-V2/V3 push MLA; Llama/Qwen stay GQA-mainline; engines add support over time.

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
