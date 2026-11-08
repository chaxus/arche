## Idea

### 1. What it is for

Attention mixes **across positions**; each position still needs its own nonlinear map—the **FFN / MLP**.  
Llama-style **SwiGLU**: three bias-free linears (gate / up / down), gated product, project back.

$$\\mathrm{FFN}(x)=W_d\\big(\\mathrm{SiLU}(W_g x)\\odot W_u x\\big)$$

### 2. Why the LLM pipeline needs it

Attention mostly “reads others.” The model also needs rich per-token transforms. SwiGLU is the common gated FFN in modern dense decoders (vs older GELU MLPs).

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Spec** | Wider hidden (`ffn_dim`); three bias-free mats |
| **Init / train** | With the rest of the net |
| **Variant** | MoE experts are often SwiGLUs (later) |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | In/out `embed_dim` | Width only in the middle |
| **Hard** | No bias (Llama contract) | Match weight layout |
| **Soft** | Gains vs GELU MLP | Full-model metrics |

### 5. What labs ship · what we do

Modern dense: SwiGLU or close gated cousins.  
`SwiGLU` in `arche/mlp.py`.

## Run

```bash
python chapters/01-inference/ep07_swiglu.py
```

## Next

**Transformer block** (`?c=block`): the sandwich.
