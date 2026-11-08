## In one sentence
Prefill the prompt once, then decode new tokens while reusing stored keys and values.

## Before you start
- Generation chapter done. Time: ~15 minutes.

## The idea (plain language)
Without a cache, every new token re-reads the whole prefix in attention—wasteful.
With a **KV cache**: compute K/V for the prompt once, then for each new token only append its K/V and attend against the past.

## In the code
- `arche/cache.py`, used by attention + `generate(..., use_cache=True)`
- `chapters/01-inference/ep12_kv_cache.py`

## Do this now
```bash
python chapters/01-inference/ep12_kv_cache.py
```

## If something breaks
- Wrong `start_pos` after edits → restore the chapter script.

## Don't worry about yet
Paged KV (ep23) and production schedulers.

## Next
**RoPE scaling** (`?c=rope-scaling`).
