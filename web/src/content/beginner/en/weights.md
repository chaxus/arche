## Idea

### 1. Why load weights

Right graph + random weights = nonsense text.

### 2. What we do

`from_pretrained` builds our modules, then copies HF Llama 3.2 tensors via a **key map**.

### 3. Pitfalls

| Issue | Meaning |
| --- | --- |
| Wrong tokenizer | Ids won’t match rows (see tokenizer chapter) |
| Missing gated access | Can’t download |
| Key map bugs | Silent wrong layers |

### 4. What labs ship

Official HF Llama 3.2 checkpoints (license + login).

### 5. What “good” means

- Param counts look right  
- Later HF align chapter: logits close on a short prompt  

### 6. What we do

`arche/load_weights.py` + chapter script.

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## Fix

Same gated issues as tokenizer: license, login, mirror.

## Next

**Generate** (`?c=generate`).
