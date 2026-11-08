## Idea

From zero, huge capacity at controlled compute often means **MoE**—sparse experts (Mixtral, DeepSeek-MoE, …).

### 1. What it is for

Many SwiGLU experts + router; each token runs top-k only.

### 2. From zero: how you build it

Pick E/top-k; experts + router; aux load-balance; sparse infer; shard experts in checkpoints. This lesson is an API primer.

### 3. What “good” means

Valid routes; correct merge; balanced load; quality/capacity trade OK.

### 4. Improve · whole-model impact

More experts/k ⇒ capacity/compute; balance loss fights collapse; replacing dense FFN is architecture-level; multi-GPU expert parallel is the engineering cost.

### 5. Industry · open models

Mixtral top-2; DeepSeek-V3 large MoE; Qwen-MoE variants; Llama 3.2 1B specimen stays dense first.

### 6. This course

```bash
python chapters/01-inference/ep14_moe.py
```

## Run

```bash
python chapters/01-inference/ep14_moe.py
```

## Next

**MLA** (`/mla/`).
