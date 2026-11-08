## Idea

### 1. What it is for

**RMSNorm** rescales each token’s float row: compute RMS, divide by it, multiply by a learnable `γ`. Unlike LayerNorm, it **does not subtract the mean**.

In Llama-style stacks it runs before attention and before the MLP (**pre-norm**).

### 2. Why the LLM pipeline needs it

Deep stacks blow up or vanish without scale control. Norm is the “re-zero the ruler” step before the next heavy op. Without it, attention/MLP are hard to stack stably.

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Spec** | Normalize the last dim; `γ` length = model dim |
| **Init** | `γ` often starts at 1 |
| **Train** | Gradients update `γ` with the rest of the net |
| **Infer** | Apply the formula; shape unchanged |

$$\mathrm{RMSNorm}(x)=x/\sqrt{\mathrm{mean}(x^2)+\varepsilon}\odot\gamma$$

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Same shape in/out | Don’t change T or dim |
| **Hard** | `ε` present | Avoid div-by-zero |
| **Hard** | Agreed placement | This course: pre-norm |
| **Soft** | With `γ=1`, per-token RMS ≈ 1 | Sanity that the formula is right |
| **Soft** | Stabler training | Final judge is full-model loss |

### 5. What labs ship · what we do

Llama 3.2 / Qwen dense: RMSNorm + pre-norm.  
`RMSNorm` in `arche/norms.py`; lab prints shapes and per-token RMS.

## Run

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## Next

**RoPE** (`?c=rope`): write position into Q/K.
