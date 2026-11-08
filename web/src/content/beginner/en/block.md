## Idea

One **Transformer block** is the repeatable sandwich:

```
x → RMSNorm → Attention → + residual
  → RMSNorm → SwiGLU   → + residual
```

Same shape in and out, so you stack it `N` times (Llama 3.2 1B: **16** layers). Pre-norm (normalize *before* the sublayer) is what modern dense models use; residuals keep gradients alive.

Optional KV cache threads through attention for decode (you will use it soon).

## Run

```bash
python chapters/01-inference/ep08_block.py
```

Expect one block forward; shapes unchanged across the residual path.

## Next

**Stack + LM head** (`?c=stack`): embed → many blocks → vocab scores.
