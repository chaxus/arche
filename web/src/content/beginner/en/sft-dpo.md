## Idea

**What:** **Alignment after pretrain** — SFT teaches “answer like the demos”; DPO (and RLHF-family methods) push **chosen vs rejected** preferences without a full RL loop in this lesson.

**Why:** Next-token pretrain makes a base LM; it does not by itself follow instructions or match product prefs. Align steers an already capable net—it does **not** invent the base skill.

**How:** Instruction data → mask-prompt SFT from a base ckpt → preference pairs → DPO (or RLHF) → eval regressions.

### 1. What — SFT loss

[`sft_loss`](https://github.com/chaxus/arche/blob/main/arche/train/sft.py): same shift as pretrain, but **only response tokens** get CE:

```python
# shift_logits vs shift_labels as usual
mask[i, start:] = True  # start ≈ prompt_len - 1
return losses[mask].mean()
```

Prompt tokens are context, not supervised targets—otherwise the model “learns to rewrite the user.”

### 2. What — DPO sketch

[`dpo_loss`](https://github.com/chaxus/arche/blob/main/arche/train/dpo.py): policy vs frozen **reference**; prefer chosen completion over rejected (β-scaled).

| Stage | Data | Objective |
| --- | --- | --- |
| **SFT** | prompt → good reply | CE on reply span |
| **DPO** | prompt + chosen/rejected | Prefer chosen logprob margin vs ref |
| **RLHF** (map only) | reward model + PPO-like | Not coded here |

### 3. Why — order and risks

| Choice | Why |
| --- | --- |
| After pretrain | Need a next-token-capable base |
| Mask prompts | Don’t burn loss on user text |
| Ref model in DPO | Anchor so policy doesn’t drift unbound |
| Data > blind steps | Bad demos teach bad habits |

Over-align can dull base capability; often paired with new specials / chat templates.

### 4. How — from zero

| Step | Do |
| --- | --- |
| 1 | Start from base (or instruct) checkpoint |
| 2 | Build prompt/response ids + `prompt_lens` |
| 3 | `train_sft_step` / `train_dpo_step` toys |
| 4 | Eval following + regressions before ship |

### 5. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Correct masks/shapes; ref frozen in DPO |
| **Soft** | Better following / safety; limited capability loss |
| **Soft** | Template + tokenizer stay paired |

### 6. This course

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## Run

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## Next

**KV cache** (`/kv-cache/`): make decode affordable before serving.
