## Idea

From-zero serving last mile: **paged KV** fights fragmentation; **CUDA Graph** cuts fixed-shape launch tax—a map toward vLLM-class engines.

### 1. What it is for

Block table + gather; optional CUDA Graph capture. Teaching ≠ drop-in vLLM.

### 2. From zero: how you build it

Logical KV + continuous batch → block alloc/free → optional graph on decode.

### 3. What “good” means

Gather matches logical cache; in-bounds tables; less fragmentation/launch; higher throughput.

### 4. Improve · whole-model impact

Page size/prealloc change waste vs speed; graphs need stable shapes; doesn’t change weights—raises serving ceiling.

### 5. Industry · open models

vLLM paged attention landmark; TRT-LLM/SGLang variants; pairs with Llama/Qwen/DeepSeek weights.

### 6. This course

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## Run

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## Next


**MoE** (`?c=moe`): frontier architectures.
