## Idea

Serving many users naively = one generate loop each (GPUs idle between people).

**Continuous batching** (teaching scheduler here): keep several short decode jobs in one stepping loop; when a job hits max new tokens (or EOS), retire it and optionally admit another. Everyone shares the same “time step” rhythm.

This is **not** full paged attention / vLLM—just the scheduling shape. Try the demo on this page.

## Run

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

Expect several jobs advancing together until each finishes.

## Next

**Pretrain** (`?c=pretrain`): a tiny next-token training loop.
