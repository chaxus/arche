## In one sentence
RMSNorm keeps layer inputs well-scaled without the extra mean-centering of LayerNorm.

## Before you start
- Embedding done. Time: ~12 minutes.

## The idea (plain language)
Deep stacks multiply numbers again and again. Without normalization, values can explode or vanish.
**RMSNorm** divides by a root-mean-square size, then scales with a learned gain. It skips subtracting the mean—cheaper, still stable enough for modern LLMs.
Inside each block it runs **before** attention and **before** the MLP (pre-norm).

## Picture / formula
$$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\mathrm{mean}(x^2)+\varepsilon}} \odot \gamma$$

## In the code
- `arche/norms.py`
- `chapters/01-inference/ep04_rmsnorm.py`

## Do this now
```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## If something breaks
- NaNs in experiments usually mean epsilon or inputs were edited—restore the script.

## Don't worry about yet
Every LayerNorm variant in the literature.

## Next
**RoPE** (`?c=rope`).
