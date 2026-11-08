## Idea

**Alignment after pretrain**: SFT teaches the model to answer like your demos; DPO (and RLHF-family methods) nudge **chosen vs rejected** preferences without a full RL loop in this lesson. Next-token pretrain gives you a capable base LM—it does not by itself follow instructions or match product taste. Alignment steers an existing net; it does not invent the base skill. Flow: instruction data → mask-prompt SFT from a base ckpt → preference pairs → DPO → regression evals.

### SFT loss

[`sft_loss`](https://github.com/chaxus/arche/blob/main/arche/train/sft.py): same shift as pretrain, but **only response tokens** get CE:

```python
# shift_logits vs shift_labels as usual
mask[i, start:] = True  # start ≈ prompt_len - 1
return losses[mask].mean()
```

Prompt tokens are context, not supervised targets—otherwise the model “learns to rewrite the user.”

### DPO sketch

[`dpo_loss`](https://github.com/chaxus/arche/blob/main/arche/train/dpo.py): policy vs frozen **reference**; prefer chosen completion over rejected (β-scaled).

| Stage | Data | Objective |
| --- | --- | --- |
| **SFT** | prompt → good reply | CE on reply span |
| **DPO** | prompt + chosen/rejected | Prefer chosen logprob margin vs ref |
| **RLHF** (map only) | reward model + PPO-like | Not coded here |

### Order and risks

| Choice | Why it matters |
| --- | --- |
| After pretrain | Need a next-token-capable base |
| Mask prompts | Don’t burn loss on user text |
| Ref model in DPO | Anchor so policy doesn’t drift unbound |
| Data > blind steps | Bad demos teach bad habits |

Over-align can dull base capability; often paired with new specials / chat templates.

### Run a toy training step

| Step | Do |
| --- | --- |
| 1 | Start from base (or instruct) checkpoint |
| 2 | Build prompt/response ids + `prompt_lens` |
| 3 | `train_sft_step` / `train_dpo_step` toys |
| 4 | Eval following + regressions before ship |

### Check yourself

| Must hold | Correct masks/shapes; ref frozen in DPO |
| Worth aiming for | Better following / safety; limited capability loss |
| Worth aiming for | Template + tokenizer stay paired |

### This course

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## Run

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## Next

**KV cache** (`/kv-cache/`): make decode affordable before serving.
