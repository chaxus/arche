## Idea

### 1. Why normalize

Deep stacks blow up or die if every layer sees wild scales. Normalization keeps vectors usable.

### 2. What RMSNorm does

For each token vector (last dim):

1. Compute RMS  
2. Divide by RMS (+ tiny `ε`)  
3. Multiply by learnable `γ`  
4. **No mean subtract** (unlike LayerNorm)

We use **pre-norm**: before attention and before MLP.

$$\mathrm{RMSNorm}(x)=x/\sqrt{\mathrm{mean}(x^2)+\varepsilon}\odot\gamma$$

### 3. Names you will hear

| Name | Plain idea | Today |
| --- | --- | --- |
| LayerNorm | Mean-center + scale | Older GPT-2-style stories |
| **RMSNorm** | Scale by RMS only | **Llama / Qwen default** |
| Pre-norm vs post-norm | Norm before vs after sublayer | Modern stacks: pre-norm |

### 4. What labs ship

Llama 3.2, Qwen dense, most new open decoders: RMSNorm + pre-norm.

### 5. What “good” means

- Same shape in/out  
- With `γ=1`, post-norm RMS ≈ 1  
- Stable training without being the whole story (optimizer/data still matter)  

### 6. What we do

`RMSNorm` in `arche/norms.py`. Lab prints shapes and per-token RMS.

## Run

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## Next

**RoPE** (`?c=rope`): bake position into Q/K.
