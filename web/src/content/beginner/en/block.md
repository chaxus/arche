## Idea

From zero: parts become one **repeatable layer** so you can stack depth.

### 1. What it is for

Pre-norm sandwich: RMSNorm→attn→res→RMSNorm→SwiGLU→res. Same shape in/out.

### 2. From zero: how you build it

Wire the sandwich; per-block weights; pick depth N (Llama 3.2 1B = **16**); stack and train.

### 3. What “good” means

Residuals add; pre-norm placement; isolated caches; still trainable when deep.

### 4. Improve · whole-model impact

Deeper N ⇒ capability/cost; post→pre-norm enables deep from-zero training; parallel attn∥MLP isn’t Llama’s main line.

### 5. Industry · open models

Llama/Qwen: repeat pre-norm blocks. DeepSeek variants change insides; block boundary remains.

### 6. This course

```bash
python chapters/01-inference/ep08_block.py
```

## Run

```bash
python chapters/01-inference/ep08_block.py
```

## Next

**Stack + LM head** (`?c=stack`).
