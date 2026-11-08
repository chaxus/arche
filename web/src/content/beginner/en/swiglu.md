## Idea

### 1. Why an MLP after attention

Attention mixes tokens; each position still needs a **per-token** transform.

### 2. What SwiGLU does

$$\mathrm{FFN}(x)=W_d\big(\mathrm{SiLU}(W_g x)\odot W_u x\big)$$

Three bias-free linears: **gate**, **up**, **down**.

### 3. Alternatives

| FFN | Plain idea | Today |
| --- | --- | --- |
| GELU MLP (GPT-2 style) | One hidden width + GELU | Older teaching fossils |
| **SwiGLU** | Gated SiLU × up | **Llama / Qwen dense** |
| MoE of SwiGLU experts | Router picks top-k | Mixtral-style (later) |

### 4. What labs ship

Modern dense decoders: SwiGLU (or close gated variants).

### 5. What “good” means

- In/out `embed_dim`; intermediate wider  
- No bias on the three linears (Llama-style)  

### 6. What we do

`SwiGLU` in `arche/mlp.py`.

## Run

```bash
python chapters/01-inference/ep07_swiglu.py
```

## Next

**Block** (`?c=block`): glue the sandwich.
