## Idea

### 1. What it is for

**Pretraining** teaches next-token prediction on lots of text: logits → cross-entropy vs the next id → update weights (including the embedding table). This is where most trunk parameters are born.

### 2. Why the LLM pipeline needs it

Random weights can’t speak. Learn statistics/structure on huge text first; then chat and preferences. Without pretrain, SFT/DPO have no real starting point.

### 3. Where it comes from · how you build it

| Scale | Meaning |
| --- | --- |
| **This lab** | Toy AdamW: few steps, feel the API |
| **Page playground** | Drag lr/steps; watch loss |
| **Real pretrain** | Massive data, multi-GPU, long runs |

Intuition: batch → predict next token → loss → `optimizer.step()`.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Loss is next-id CE | Right objective |
| **Hard** | Grads hit the right params | Including embeddings |
| **Soft** | Loss trends down | Enough on the toy |
| **Soft** | Real runs: data, stability, downstream | Not claimed here |

### 5. What labs ship · what we do

`arche/train/pretrain.py` + page playground. Not internet-scale training.

## Run

```bash
python chapters/02-training/ep18_pretrain.py
```

## Next

**SFT · DPO** (`?c=sft-dpo`).
