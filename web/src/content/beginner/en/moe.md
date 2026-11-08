## Idea

From zero, huge capacity at controlled compute often means **MoE**—sparse experts. **DeepSeek-V3**’s open story is largely **MoE (capacity) + MLA (KV)**; this chapter is the MoE half. Mixtral is another open MoE; our Llama 3.2 example stayed **dense** on purpose.

### 1. What it is for

Many SwiGLU experts + router; each token runs top-k only. Replaces the dense FFN you wrote earlier—same decoder spine, different capacity trick.

### 2. From zero: how you build it

Pick E/top-k; experts + router; aux load-balance; sparse infer; shard experts in checkpoints. This lesson is an API primer.

### 3. What “good” means

Valid routes; correct merge; balanced load; quality/capacity trade OK.

### 4. Improve · whole-model impact

More experts/k ⇒ capacity/compute; balance loss fights collapse; replacing dense FFN is architecture-level; multi-GPU expert parallel is the engineering cost.

### 5. Industry · open models

| Line | Role |
| --- | --- |
| **DeepSeek-V3** | Large **MoE** + **MLA** (next chapter)—flagship open non-dense stack |
| **Mixtral** | MoE with top-2 SwiGLU experts |
| **Qwen-MoE** | Own E/k recipes |
| **Llama 3.2 ~1B (this course)** | Dense first—then you can read DeepSeek’s map |

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
