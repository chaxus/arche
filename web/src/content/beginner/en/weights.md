## Idea

So far the graph was right but weights were random—output is nonsense.

**`from_pretrained`** builds our `Llama` modules, then copies tensors from the official Hugging Face Llama 3.2 checkpoint using a **key map** (HF names → arche names). After that, forward math matches the released model (up to numerics).

Needs accepting the model license and `huggingface-cli login` (same gate as the tokenizer).

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

Expect confirmation that weights landed (parameter counts / a short sanity print). Mirror tip: `export HF_ENDPOINT=https://hf-mirror.com`.

## Fix

Same gated-model issues as the tokenizer chapter: license, login, network.

## Next

**Generate** (`?c=generate`): turn logits into text, token by token.
