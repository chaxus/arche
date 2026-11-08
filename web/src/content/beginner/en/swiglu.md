## In one sentence
SwiGLU is the modern feed-forward block: gate × up, then down—three bias-free linears.

## Before you start
- GQA done. Time: ~12 minutes.

## The idea (plain language)
Old tutorials often show Linear → GELU → Linear. Llama-style models prefer:
`down(silu(gate(x)) * up(x))`.
The **gate** decides what to pass; **up** proposes values; **down** returns to model width.

## Picture / formula
$$\mathrm{FFN}(x)=W_d\big(\mathrm{SiLU}(W_g x)\odot W_u x\big)$$

## In the code
- `arche/mlp.py`
- `chapters/01-inference/ep07_swiglu.py`

## Do this now
```bash
python chapters/01-inference/ep07_swiglu.py
```

## If something breaks
- Shape mismatch → intermediate_dim must match the three linears in the demo.

## Don't worry about yet
MoE (many experts)—that is Part 2.

## Next
**Transformer block** (`?c=block`).
