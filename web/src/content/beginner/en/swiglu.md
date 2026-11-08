## Idea

From-zero blocks: attention mixes positions; **SwiGLU FFN** transforms each position—modern dense default.

### 1. What it is for

Gated SiLU FFN with three bias-free linears (gate/up/down).

### 2. From zero: how you build it

Pick `ffn_dim`, three mats, wire after second pre-norm; MoE experts are often SwiGLUs.

### 3. What “good” means

In/out `D`; no bias (Llama contract); judge gains via full-model metrics.

### 4. Improve · whole-model impact

Wider FFN ⇒ capacity/cost; GELU→SwiGLU needs retrain; MoE multiplies experts.

### 5. Industry · open models

Llama/Qwen dense: SwiGLU. Mixtral-style MoE: SwiGLU experts. GPT-2 teaching: GELU MLP.

### 6. This course

```bash
python chapters/01-inference/ep07_swiglu.py
```

## Run

```bash
python chapters/01-inference/ep07_swiglu.py
```

## Next

**Transformer block** (`?c=block`).
