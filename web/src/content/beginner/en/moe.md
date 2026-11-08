## Idea

### 1. What it is for

**MoE** = many expert FFNs (here SwiGLU) + a **router**: each token runs only **top-k** experts—sparse compute for more capacity.

### 2. Why the LLM pipeline needs it

Dense MLPs charge every token full width. MoE aims for huge total params while each token pays for only a few experts (at the cost of harder train/serve).

### 3. Where it comes from · how you build it

| Step | What happens |
| --- | --- |
| **1** | N experts + router |
| **2** | Score tokens; pick top-k |
| **3** | Run chosen experts; weighted merge |
| **Train** | Often load-balancing aux losses |

This lesson is a teaching MoE API—not a full Mixtral stack.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Valid expert indices | No OOB routes |
| **Hard** | top-k matches impl | Shapes/merge correct |
| **Soft** | Load not extremely skewed | Critical in real training |
| **Soft** | Quality vs compute trade | Product evals |

### 5. What labs ship · what we do

Mixtral-style top-k; shared-expert variants exist.  
Primer in `arche/moe.py`.

## Run

```bash
python chapters/01-inference/ep14_moe.py
```

## Next

**MLA** (`?c=mla`).
