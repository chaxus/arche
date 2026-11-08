## Idea

### 1. What it is for

Post-pretrain alignment APIs in this lesson:

- **SFT**: prompt→answer demos; **mask the prompt**; loss only on reply tokens  
- **DPO**: vs a frozen reference, prefer chosen over rejected (one teaching step)

### 2. Why the LLM pipeline needs it

Pretrain completes web text; it doesn’t automatically “be an assistant.” SFT teaches format/following; DPO/RLHF-style methods push toward preferences. Chat products need some of this.

### 3. Where it comes from · how you build it

| Method | Data | Goal |
| --- | --- | --- |
| SFT | instruction–reply pairs | Imitate replies |
| DPO | chosen/rejected for a prompt | Preference (no explicit RM in this recipe) |
| Broader map | RLHF / RLAIF / … | Many recipes; two APIs here |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | SFT mask correct | No loss on the prompt |
| **Hard** | DPO has reference + pairs | API shape |
| **Soft** | Better instruction-following | Human/auto evals |
| **Soft** | Don’t erase pretrain skill | Data/lr tradeoffs |

### 5. What labs ship · what we do

`arche/train/sft.py` (and DPO) + this script.

## Run

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## Next

**Frontier checkpoints** (`?c=frontier-ckpt`).
