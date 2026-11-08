## Idea

Full multi-head attention: every query head has its own key/value head. That makes the **KV cache** (next chapters) large at decode time.

**GQA** (Grouped-Query Attention): fewer KV heads than query heads. Several query heads **share** one KV head. Llama 3.2 1B uses **32 query / 8 KV** heads—same quality ballpark, smaller cache.

Causal mask still applies: position `t` only sees positions `≤ t`.

Flow in our code: project Q/K/V → reshape heads → RoPE on Q/K → expand KV with `repeat_interleave` → scaled dot-product attention → output projection.

## Run

```bash
python chapters/01-inference/ep06_gqa.py
```

Expect a tiny forward that prints head counts and output shape. Use the interactive diagram on this page.

## Next

**SwiGLU** (`?c=swiglu`): the feed-forward half of each block.
