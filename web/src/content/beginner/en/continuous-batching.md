## Idea

### 1. Why

One generate loop per user wastes GPU when people finish at different times.

### 2. What continuous batching is here

Several short decode jobs share one stepping loop; finished jobs retire. **Not** full paged attention / vLLM. Try the demo.

### 3. Landscape

| Layer | Role |
| --- | --- |
| Scheduling | Who is active this step |
| Paging / kernels | How memory & compute run (later primer) |

### 4–6. What we do

Teaching scheduler shape in the chapter script.

## Run

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## Next

**Pretrain** (`?c=pretrain`).
