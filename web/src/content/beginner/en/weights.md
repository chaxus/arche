## Idea

From zero, two paths meet: **(A) write your own checkpoint via pretrain**; **(B) load an open checkpoint**. Bad loads waste the handwritten stack.

### 1. What it is for

Copy tensors via a key map. Random weights are a correct graph with nonsense outputs.

### 2. From zero: how you do it

Self-train: save state_dict + tokenizer version. Open load: license→download→key map→official tokenizer→validate.

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
