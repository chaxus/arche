## Idea

**RMSNorm** keeps each token’s activations on a stable scale: divide by root-mean-square, then multiply a learnable `γ`. There is no mean subtraction and no bias. Deep next-token stacks blow up or collapse without that steadying step; Llama-style **pre-norm** (normalize before attention and MLP) is what makes depth trainable from scratch. In practice you place `RMSNorm(D)` at each pre-norm site, initialize `γ ≈ 1`, and train it with the rest of the net.

### Formula and code

\[
\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\mathrm{mean}(x^2)+\varepsilon}}\odot\gamma
\]

Repo ([`arche/norms.py`](https://github.com/chaxus/arche/blob/main/arche/norms.py)):

```python
variance = x_f.pow(2).mean(dim=-1, keepdim=True)
x_normed = x_f * torch.rsqrt(variance + self.eps)
return (self.weight.float() * x_normed).to(orig_dtype)
```

Math runs in float32, then casts back—the same stability trick HF Llama uses.

### Compared with LayerNorm and placement

| Choice | Rationale |
| --- | --- |
| No mean center | Slightly cheaper; Llama/Qwen recipe |
| Learnable `γ` only | Scale per channel after unit-RMS |
| **Pre-norm** | Stabilize *inputs* to attn/MLP so depth survives |
| After embed, before bricks | First “steady the ruler” once vectors exist |

GPT-2-era teaching often used LayerNorm + post-norm. This course’s example is **RMSNorm + pre-norm**.

### Wiring it in

| Step | Do |
| --- | --- |
| 1 | `RMSNorm(D)` before attention and before SwiGLU (see block) |
| 2 | Final RMSNorm before LM head (see stack) |
| 3 | Match `ε` to the checkpoint (`norm_eps`) |
| 4 | Train / load; don’t swap LN↔RMSN under foreign weights |

### Check yourself

- Output shape matches input; `ε` is set; pre-norm sites match the recipe.
- With `γ=1`, per-token RMS ≈ 1; fewer NaNs in a deep forward.
- Full-model loss and HF align are the real judge.

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| LN → RMSN | Architecture change ⇒ retrain |
| Pre vs post | Depth trainability |
| `γ` / lr | Couples to convergence |

### This course

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## Run

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## Next

**RoPE** (`/rope/`): write position into Q/K.
