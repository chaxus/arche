## Idea

**What:** The **optimizer side** of one pretrain step—AdamW, learning rate (and schedules), gradient clipping, and gradient accumulation—so the CE objective actually moves weights stably.

**Why:** Correct loss + bad optim → NaNs, divergence, or “loss stuck.” At any scale past toys, **how you step** decides whether a run finishes.

**How:** `zero_grad` → forward → loss → `backward` → (optional clip) → `optimizer.step` → (optional lr schedule). Accumulate grads over micro-batches when `B` must stay small.

### 1. What — knobs

| Knob | Role |
| --- | --- |
| **AdamW** | Adaptive moments + **decoupled weight decay** (default in Llama/nanoGPT-style recipes) |
| **lr** | Step size; too high → explode; too low → crawl |
| **schedule** | Warmup then decay (cosine / WSD / …) so early steps don’t thrash random embeds |
| **grad clip** | Cap global grad norm (e.g. 1.0) against spikes |
| **grad accum** | Sum grads over `K` micro-batches ≈ larger effective batch |

This repo’s toy loop ([`train_pretrain_steps`](https://github.com/chaxus/arche/blob/main/arche/train/pretrain.py)):

```python
opt = torch.optim.AdamW(model.parameters(), lr=lr)
for batch in batches:
    opt.zero_grad(set_to_none=True)
    logits, _ = model(batch)
    loss = lm_loss(logits, batch)
    loss.backward()
    opt.step()
```

No schedule/clip yet—on purpose. Production adds them around the same skeleton.

### 2. Why — each piece

| Choice | Why |
| --- | --- |
| AdamW over plain SGD | Sparse-ish token grads + many params; adaptive scales help |
| Weight decay | Mild regularization; AdamW’s form matches LLM practice |
| Warmup | Early CE is huge (`~log V`); blunt lr shreds init |
| Clip | One bad batch shouldn’t blow FP math |
| Accum | Fit long `T` / big models on limited VRAM while keeping effective batch |

### 3. How — accumulation sketch

```python
opt.zero_grad(set_to_none=True)
for micro in micro_batches:
    loss = lm_loss(*model_and_shift(micro)) / K
    loss.backward()          # grads accumulate
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
# scheduler.step()  # if any
```

```mermaid
flowchart LR
  mb[micro_batches] --> bwd[backward_x_K]
  bwd --> clip[clip_grad_norm]
  clip --> step[AdamW_step]
  step --> sch[lr_schedule]
```

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | No NaN/Inf in loss or grads after clip |
| **Hard** | `zero_grad` each **optimizer** step (not forgotten under accum) |
| **Soft** | Loss trends down over a sensible window |
| **Soft** | lr schedule matches a known recipe when you scale up |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| lr / warmup | Trainability of deep stacks + embeds |
| effective batch (B×accum×world) | Noise vs stability of updates |
| clip threshold | Spike survival vs under-updating |
| Wrong decay | Over-shrink or under-regularize weights |

### 6. This course

```bash
python chapters/02-training/ep18_pretrain.py
```

Page demo: try higher lr until loss misbehaves—that’s the optim lesson in miniature.

## Run

```bash
python chapters/02-training/ep18_pretrain.py
```

## Next

**Pretrain scale** (`/pretrain-scale/`): how frontier recipes stretch the same loop.
