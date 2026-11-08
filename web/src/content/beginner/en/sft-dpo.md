## Idea

After from-zero pretrain you still **align**: SFT teaches following; DPO/RLHF-style methods push preferences.

### 1. What it is for

SFT mask-prompt loss on replies; DPO prefer chosen vs rejected against a reference.

### 2. From zero: how you build it

Instruction/preference data → SFT from base ckpt → DPO/RLHF → eval regressions.

### 3. What “good” means

Correct masks/shapes; better following/safety; limited capability loss.

### 4. Improve · whole-model impact

Data quality beats blind steps; over-align can dull the base; often paired with new specials/templates.

### 5. Industry · open models

Llama/Qwen Instruct/Chat; Zephyr-style DPO; TRL-like tooling; labs use RLHF variants.

### 6. This course

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## Run

```bash
python chapters/02-training/ep19_sft_dpo.py
```

## Next


**KV cache** (`?c=kv-cache`): make decode affordable before serving.
