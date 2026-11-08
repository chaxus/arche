## Idea

GQA already shrinks KV heads. **MLA** (Multi-head Latent Attention, DeepSeek-style idea) goes further for decode memory: cache a **low-rank latent** (plus a small RoPE stream) instead of full per-head K and V, then reconstruct what attention needs on the fly.

Teaching goal: feel *why* latent KV helps memory—not ship a bit-identical DeepSeek checkpoint.

## Run

```bash
python chapters/01-inference/ep15_mla.py
```

Expect a compact forward that shows latent cache shapes.

## Next

**Device · INT8 · chat** (`?c=device-quant-chat`): where it runs, how weights shrink, how chats are formatted.
