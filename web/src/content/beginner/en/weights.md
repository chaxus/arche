## Idea

Still one path from zero. This chapter loads the **Llama example’s** open checkpoint so your handwritten stack can speak now. Part 2 is where you feel how those floats get **written by training**—most numbers are trained, not hand-filled. Bad loads waste the handwritten stack.

### 1. What it is for

Copy tensors via a key map. Random weights are a correct graph with nonsense outputs; Llama 3.2 weights make the same forward pass behave like a language model.

### 2. From zero: how you do it

| Step on the same path | What you do |
| --- | --- |
| **Load the example (this chapter)** | license→download→key map→official tokenizer |
| **Train later (Part 2)** | toy/real pretrain writes `state_dict`; record tokenizer version |
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

**Text generation** (`/generate/`).
