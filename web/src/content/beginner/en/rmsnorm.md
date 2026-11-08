## Idea

From-zero deep nets need scale control after embeddings—RMSNorm is the first “steady the ruler.”

### 1. What it is for

**RMSNorm**: divide by RMS, multiply learnable `γ`; **no mean subtract**. Llama-style **pre-norm** before attention and MLP.

### 2. From zero: how you build it

| Step | What you do |
| --- | --- |
| **1** | `RMSNorm(D)` at each entry that needs steadying |
| **2** | init `γ` ≈ 1 |
| **3** | wire pre-norm before Attention/MLP |
| **4** | train with the net |

### 3. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | same shape; `ε`; placement matches recipe |
| **Soft** | with `γ=1`, RMS≈1; fewer NaNs |
| **Soft** | full-model loss is the judge |

### 4. Improve · whole-model impact

LN→RMSN matches Llama/Qwen; pre- vs post-norm decides depth trainability; `γ`/lr couple to convergence.

### 5. Industry · open models

Llama 3/3.2 & Qwen: RMSNorm+pre-norm. GPT-2-era teaching stacks: LayerNorm often post-norm.

### 6. This course

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## Run

```bash
python chapters/01-inference/ep04_rmsnorm.py
```

## Next

**RoPE** (`/rope/`): write position into Q/K from zero.
