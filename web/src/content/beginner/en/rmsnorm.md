## Idea

**What:** **RMSNorm** — per-token scale control: divide by root-mean-square, multiply learnable `γ`. No mean subtract, no bias.

**Why:** Deep next-token stacks blow up or die without steady scales. Llama-style **pre-norm** (norm → attn / MLP) is what makes depth trainable from zero.

**How:** `RMSNorm(D)` at each pre-norm site; `γ ≈ 1` at init; train with the net.

### 1. What — formula

\[
\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\mathrm{mean}(x^2)+\varepsilon}}\odot\gamma
\]

Repo ([`arche/norms.py`](https://github.com/chaxus/arche/blob/main/arche/norms.py)):

```python
variance = x_f.pow(2).mean(dim=-1, keepdim=True)
x_normed = x_f * torch.rsqrt(variance + self.eps)
return (self.weight.float() * x_normed).to(orig_dtype)
```

Math in float32, cast back—same stability trick HF Llama uses.

### 2. Why — vs LayerNorm / placement

| Choice | Why |
| --- | --- |
| No mean center | Slightly cheaper; Llama/Qwen recipe |
| Learnable `γ` only | Scale per channel after unit-RMS |
| **Pre-norm** | Stabilize *inputs* to attn/MLP so depth survives |
| After embed, before bricks | First “steady the ruler” once vectors exist |

GPT-2-era teaching often: LayerNorm + post-norm. This course’s example is **RMSNorm + pre-norm**.

### 3. How — wire it

| Step | Do |
| --- | --- |
| 1 | `RMSNorm(D)` before attention and before SwiGLU (see block) |
| 2 | Final RMSNorm before LM head (see stack) |
| 3 | Match `ε` to the checkpoint (`norm_eps`) |
| 4 | Train / load; don’t swap LN↔RMSN under foreign weights |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Same shape as input; `ε` present; pre-norm sites match recipe |
| **Soft** | With `γ=1`, RMS≈1; fewer NaNs in deep forward |
| **Soft** | Full-model loss / HF align is the real judge |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| LN → RMSN | Architecture change ⇒ retrain |
| Pre vs post | Depth trainability |
| `γ` / lr | Couples to convergence |

### 6. This course

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## Run

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## Next

**RoPE** (`/rope/`): write position into Q/K.
