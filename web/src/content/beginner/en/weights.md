## In one sentence
Map official Hugging Face Llama 3.2 tensors into the hand-written `arche.Llama` graph.

## Before you start
- Stack chapter done. Time: ~20 minutes.
- Needs Hugging Face login + license acceptance (see Part 0 map).

## The idea (plain language)
Your modules are empty until filled. `from_pretrained` builds our graph, downloads HF weights, and copies tensors with a small key map (`model.layers.*` → `layers.*`).

## In the code
- `arche/load_weights.py`
- `chapters/01-inference/ep10_load_weights.py`

## Do this now
```bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
```

## If something breaks
- Gated errors → accept license, re-login; try `HF_ENDPOINT=https://hf-mirror.com` if needed.
- Out of disk/RAM → close other apps; 1B is the intended size.

## Don't worry about yet
Quantized serving formats or multi-GPU sharding.

## Next
**Text generation** (`?c=generate`).
