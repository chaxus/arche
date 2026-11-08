## In one sentence
Grouped-Query Attention shares key/value heads across several query heads to shrink decode-time memory.

## Before you start
- RoPE idea clear. Time: ~18 minutes.

## The idea (plain language)
Classic multi-head attention gives every query head its own K/V. That is accurate but the **KV cache** gets large while generating.
**GQA** uses fewer KV heads; each KV head is shared by a group of query heads. Llama 3.2 1B: 32 query heads, 8 KV heads.
Flow: project Q/K/V → RoPE(Q,K) → repeat KV → causal attention → output projection.

## In the code
- `arche/attention.py`
- `chapters/01-inference/ep06_gqa.py`

## Do this now
```bash
python chapters/01-inference/ep06_gqa.py
```

## If something breaks
- Assert `num_heads % num_kv_heads == 0`—groups must divide evenly.

## Don't worry about yet
FlashAttention kernels or MLA (comes later).

## Next
**SwiGLU** (`?c=swiglu`).
