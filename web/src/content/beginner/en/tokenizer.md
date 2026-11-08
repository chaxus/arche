## Idea

Text → **token ids** via the official Llama 3.2 BPE vocab (`128256`). We don’t reinvent BPE; gated HF login is required to run the lab.

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
```

## Next

**Embedding** (`?c=embedding`).
