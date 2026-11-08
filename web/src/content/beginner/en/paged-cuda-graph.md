## Idea

### 1. What it is for

Two serving primers:

1. **Paged KV** — fixed-size blocks + block table + gather  
2. **CUDA Graph** — capture fixed-shape work when CUDA is present  

Teaching mechanisms—**not** a drop-in vLLM.

### 2. Why the LLM pipeline needs it

Toy loops are fine solo; production concurrency fragments VRAM and pays launch tax. Paging manages memory; graphs cut repeated fixed-shape launch overhead.

### 3. Where it comes from · how you build it

`arche/paged_kv.py` (and graph) shows block-table gather and optional CUDA Graph capture.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Gather matches logical cache | Paging must not change meaning |
| **Hard** | Block table in bounds | Alloc/free correct |
| **Soft** | Less fragmentation / fewer launches | Serving metrics; shape here |

### 5. What we do

This primer script. Path complete—revisit weak chapters or run HF align with weights.

## Run

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## Next

Path complete—revisit weak chapters, or run HF align when you have weights.
