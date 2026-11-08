## Idea

This chapter takes **path A (specimen)**: load an open checkpoint so the handwritten stack can speak now.  
**Path B (write weights from zero)** is Part 2 pretrain / SFT·DPO—most floats are trained, not hand-filled. Bad loads waste the handwritten stack.

### 1. What it is for

Copy tensors via a key map. Random weights are a correct graph with nonsense outputs.

### 2. From zero: how you do it

| Path | Steps |
| --- | --- |
| **A Load open (this chapter)** | license→download→key map→official tokenizer |
| **B Self-train (Part 2)** | toy/real pretrain writes `state_dict`; record tokenizer version |
| **Validate** | shapes/missing keys; **HF align** after generate compares short-prompt logits |

### 3. What “good” means

Shapes/counts; full map; tokenizer pair; optional HF logit align.

### 4. Improve · whole-model impact

Fix maps; dtype; resume carefully; wrong tokenizer destroys meaning.

### 5. Industry · open models

HF + safetensors standard; Llama/Qwen/DeepSeek ship that way; engines consume the same.

### 6. This course

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## Fix

License, login, mirror—same as tokenizer.

## Next

**Text generation** (`?c=generate`).
