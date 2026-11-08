## Idea

### 1. Why

After pretrain, models still need to follow chats and prefer better answers.

### 2. What

- **SFT:** prompt→response; **mask the prompt**; loss on reply only  
- **DPO:** prefer chosen over rejected vs a frozen reference (one teaching step)

### 3. Landscape

RLHF / RLAIF / DPO / … many alignment recipes; we taste two APIs.

### 4–6. What we do

`arche/train/sft.py` (+ DPO helper) and the chapter script.

## Run

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## Next

**Frontier checkpoints** (`?c=frontier-ckpt`).
