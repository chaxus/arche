## Idea

From-zero core: **pretrain** writes embeddings and almost all trunk weights. Without it you only have a random shell.

### 1. What it is for

Next-token CE on huge text; backprop updates the net including embeds.

### 2. From zero: how you build it

Frozen tokenizer + shell + data → AdamW next-id loss → long multi-GPU run → checkpoints. This lesson: toy steps + page playground.

### 3. What “good” means

Correct objective; grads hit params; loss/downstream/stability.

### 4. Improve · whole-model impact

Data mix dominates; scale laws; stability decides finishability; undertrain ⇒ weak embeds and deep layers.

### 5. Industry · open models

Llama/Qwen/DeepSeek: pretrain then align; Megatron/DeepSpeed/torchtitan-style stacks for from-zero training.

### 6. This course

```bash
python chapters/02-training/ep18_pretrain.py
```

## Run

```bash
python chapters/02-training/ep18_pretrain.py
```

## Next

**SFT · DPO** (`?c=sft-dpo`).
