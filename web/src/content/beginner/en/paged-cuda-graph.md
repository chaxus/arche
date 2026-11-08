## Idea

Two serving ideas, teaching-sized:

1. **Paged KV** — store KV in fixed-size **blocks** + a **block table**, gather what attention needs (vLLM-inspired shape). Stops one long sequence from needing one giant contiguous cache allocation story.  
2. **CUDA Graph** — when CUDA exists, **capture** a fixed-shape compute graph to cut launch overhead; decode may stay eager until KV writes are safely in-place.

Labs show the mechanisms; they are not a drop-in production engine.

## Run

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

Expect paged gather behavior; graph capture only if CUDA is available.

## Next

Path complete. Revisit weak chapters, or run HF align with real weights when ready.
