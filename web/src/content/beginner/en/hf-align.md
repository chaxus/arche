## Idea

From-zero handwritten stack acceptance: **match authoritative HF logits** on a short prompt.

### 1. What it is for

Same prompt through arche vs HF Llama 3.2; compare argmax / max abs diff.

### 2. From zero: how you build it

Fixed seed/dtype; same weights; short seq; clean skip without weights.

### 3. What “good” means

Close enough with weights; explicit skip without; longer coverage later.

### 4. Improve · whole-model impact

Fail ⇒ revisit maps/RoPE/norm. Pass before optimize/serve.

### 5. Industry · open models

Repro CI often gates on HF/official match; engines keep numeric regressions.

### 6. This course

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## Fix

No weights/login → skip ≠ broken logic.

## Next


**Pretrain** (`?c=pretrain`): write weights from zero (toy API).
