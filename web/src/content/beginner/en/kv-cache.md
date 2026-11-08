## Idea

### 1. What it is for

**KV cache** stores per-layer K/V after prefill. Each decode step only computes the new position’s K/V, appends, and attends over the cache—no full-prompt recompute every step.

### 2. Why the LLM pipeline needs it

Naive decode recomputes K/V over the whole prefix every step. Generation is already a loop; cache makes that loop practical. Without it, long chats are painfully slow and expensive.

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Prefill** | One forward; write K/V per layer |
| **Decode** | Append one step; attend to cache |
| **Layout** | This course: concat in time; production may page (later) |
| **Shrink** | GQA / MLA reduce what you store |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Matches full recompute (up to float noise) | Cache must not change meaning |
| **Hard** | Append per layer, no drops | One store per layer |
| **Soft** | Cost grows with **new** tokens, not full prefix each step | The win |
| **Soft** | Memory acceptable | Then GQA / paging |

### 5. What labs ship · what we do

Every production decoder caches KV.  
`KVCache` in `arche/cache.py`.

## Run

```bash
python chapters/01-inference/ep12_kv_cache.py
```

## Next

**RoPE scaling** (`?c=rope-scaling`).
