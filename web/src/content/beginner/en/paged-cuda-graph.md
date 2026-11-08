## Idea

### 1. Why

Serving needs smarter memory and lower launch overhead than a toy loop.

### 2. What

1. **Paged KV** — fixed-size blocks + block table + gather  
2. **CUDA Graph** — capture fixed-shape compute when CUDA exists  

Teaching mechanisms—not a drop-in vLLM.

### 3–6. What we do

`arche/paged_kv.py` (+ graph helper) and the chapter script.

## Run

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## Next

Path complete—revisit weak chapters or run HF align with real weights.
