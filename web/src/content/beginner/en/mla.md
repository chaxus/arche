## Idea

**MLA** caches a low-rank latent (+ small RoPE stream) instead of full per-head K/V — cheaper decode memory.

## Run

```bash
python chapters/01-inference/ep15_mla.py
```

## Next

**Device · INT8 · chat** (`?c=device-quant-chat`).
