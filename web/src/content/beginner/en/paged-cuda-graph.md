## In one sentence
Store KV in fixed pages (block table) and capture a fixed-shape CUDA graph step when CUDA exists.

## Before you start
- Continuous batching + KV cache. Time: ~18 minutes.

## The idea (plain language)
**Paged KV** allocates pages and gathers a contiguous cache for attention—vLLM-style teaching model.
**CUDA graphs** capture fixed-shape module compute; Llama decode stays eager until KV writes are in-place (pages), which engines graph next.

## In the code
- `arche/paged_kv.py`, `cuda_graphs.py`
- `chapters/01-inference/ep23_paged_cuda_graph.py`

## Do this now
```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## If something breaks
- No CUDA → graph demo skips; paged KV still runs on CPU.

## Don't worry about yet
Shipping a full vLLM replacement—this course ends by naming the production pieces.

## Next
You finished the path. Revisit weak chapters, or run real HF alignment if you have weights.
