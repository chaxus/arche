## Idea

### 1. Why pretrain

Learn next-token prediction on lots of text before following chats.

### 2. What we run

Toy AdamW loop: logits vs next id, cross-entropy, step. **On this page:** drag lr / steps and train a tiny model in-browser.

### 3. Landscape

| Scale | Meaning |
| --- | --- |
| This lab | Feel the API |
| Real pretrain | Huge data, many GPUs, weeks |

### 4–6. What we do

`arche/train/pretrain.py` + page playground. Not internet-scale training.

## Run

```bash
python chapters/02-training/ep18_pretrain.py
```

## Next

**SFT · DPO** (`?c=sft-dpo`).
