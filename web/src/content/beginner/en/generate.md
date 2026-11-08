## Idea

### 1. Why a loop

One forward gives scores for the next piece—not a full reply.

### 2. What generation does

1. Logits at the **last** position  
2. Optional temperature / top-k  
3. Greedy or sample  
4. Append; stop on EOS or length  

KV cache (next chapter) makes step 1 cheap after prefill.

### 3. Knobs

| Knob | Effect |
| --- | --- |
| temperature → 0 | Greedy / sharp |
| higher temperature | More random |
| top-k | Keep only top candidates |

### 4. What labs ship

Same loop idea in every chat stack; sampling details vary.

### 5. What “good” means

- Stops cleanly  
- Respects EOS when provided  
- With real weights, text is coherent (not this chapter’s only goal)  

### 6. What we do

`generate` in `arche/generate.py`.

## Run

```bash
python chapters/01-inference/ep11_generate.py
```

## Next

**KV cache** (`?c=kv-cache`).
