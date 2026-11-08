## Idea

Deep stacks blow up or die if every layer sees wild scales. **Normalization** keeps vectors in a sane range.

**RMSNorm** (root-mean-square norm) is the modern default here:

- For each token vector, compute RMS over the last dimension  
- Divide by that RMS (plus a tiny `ε`)  
- Multiply by a learnable scale `γ`  
- **No mean subtraction** (unlike classic LayerNorm)—cheaper, still stable enough in practice  

We use it as **pre-norm**: once before attention, once before the MLP. That is what Llama / Qwen-style dense stacks do.

$$\mathrm{RMSNorm}(x)=x/\sqrt{\mathrm{mean}(x^2)+\varepsilon}\odot\gamma$$

## Run

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

Expect: same shape in/out; per-token RMS after norm ≈ 1 when `γ` starts as ones.

## Next

**RoPE** (`?c=rope`): bake position into Q/K with rotations.
