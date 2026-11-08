## Idea

**Paged KV**: block table + gather. **CUDA graph**: capture fixed-shape compute when CUDA exists (decode stays eager until KV writes are in-place).

## Run

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## Next

Path complete — revisit weak chapters or run HF align with real weights.
