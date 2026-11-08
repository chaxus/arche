## In one sentence
Multi-head Latent Attention compresses KV into a small latent cache (DeepSeek-style).

## Before you start
- KV cache intuition. Time: ~18 minutes.

## The idea (plain language)
Instead of caching full per-head K/V, **MLA** stores a low-rank latent (+ a small RoPE stream) and expands when attending.
Same attention idea, cheaper memory at decode.

## In the code
- `arche/mla.py`
- `chapters/01-inference/ep15_mla.py`

## Do this now
```bash
python chapters/01-inference/ep15_mla.py
```

## If something breaks
- Rank/head_dim mismatches → use the chapter’s constructor args.

## Don't worry about yet
Exact DeepSeek production kernels—key maps come in Part 4.

## Next
**Device · INT8 · chat** (`?c=device-quant-chat`).
