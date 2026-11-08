## Idea

### 1. Why MoE

One dense MLP per token is simple but capacity costs compute for everyone.

### 2. What MoE does

Many expert MLPs (here SwiGLU) + a **router**: each token runs only **top-k** experts → sparse compute.

### 3. Landscape

| Idea | Note |
| --- | --- |
| Dense FFN | Always-on |
| **MoE (Mixtral-style)** | Top-k experts |
| Shared experts / deep variants | Product-specific |

### 4. What labs ship

Teaching MoE block—not a full Mixtral production stack.

### 5. What “good” means

- Router picks valid experts  
- Load isn’t insanely skewed (real training cares; we only taste the API)  

### 6. What we do

`arche/moe.py` primer.

## Run

```bash
python chapters/01-inference/ep14_moe.py
```

## Next

**MLA** (`?c=mla`).
