## Idea

After attention mixes tokens, each position still needs a **per-token MLP** to transform features.

Modern stacks use **SwiGLU** instead of a plain GELU MLP:

$$\mathrm{FFN}(x)=W_d\big(\mathrm{SiLU}(W_g x)\odot W_u x\big)$$

Three bias-free linears: **gate**, **up**, **down**. SiLU on the gate, multiply by up, then down-project. Same width ideas as Llama / Qwen dense models.

## Run

```bash
python chapters/01-inference/ep07_swiglu.py
```

Expect input/output same `embed_dim`, intermediate wider.

## Next

**Block** (`?c=block`): glue norm + attention + SwiGLU with residuals.
