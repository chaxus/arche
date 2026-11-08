## In one sentence
Text becomes integer **token ids** before the model can think in vectors.

## Before you start
- Overview done.
- Time: ~15 minutes.
- Official Llama 3.2 tokenizer is **gated** on Hugging Face—login only when you run this lab.

## The idea (plain language)
Models do not read letters the way humans do. A **tokenizer** chops text into tokens and maps each to an id in a fixed vocabulary (Llama 3.2: 128256 entries).

We use the **official** tokenizer. This course focuses on the neural trunk, not reinventing compression algorithms (BPE).

## In the code
- `arche/tokenizer.py`
- Companion: `chapters/01-inference/ep02_tokenizer.py`

## Do this now
```bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
```
Expect printed token ids for a sample string. If you skip login, continue reading—you can return later.

## If something breaks
- Gated 401/403 → accept the model license on Hugging Face, then login again.
- Slow download → optional `export HF_ENDPOINT=https://hf-mirror.com` (mainland networks).

## Don't worry about yet
Training your own tokenizer or Unicode edge cases.

## Next
**Embedding** (`?c=embedding`).
