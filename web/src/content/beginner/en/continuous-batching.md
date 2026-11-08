## Idea

**What:** **Continuous batching** — a serving **scheduler** shape: many users share decode steps; finished jobs retire; new ones admit mid-flight.

**Why:** Single-user generate wastes GPU when one sequence finishes early. Industry engines (vLLM, TGI, SGLang…) keep the device busy by stepping only **active** requests together.

**How:** Active set → pad/step actives → sample → retire EOS/max → repeat. Teaching scheduler ≠ full paged-attention engine.

### 1. What — loop

[`generate_continuous`](https://github.com/chaxus/arche/blob/main/arche/batching.py):

```python
while any(not item.done for item in live):
    active = [item for item in live if not item.done]
    # pad active rows → model(batch) → sample per row → retire finished
```

Each request carries its own temperature / top-k / max tokens. No weight change—pure scheduling.

### 2. Why — throughput

| Single loop | Continuous batch |
| --- | --- |
| One user owns the step | Many share a step |
| Idle after short finish | Retire + admit keeps GPU full |
| Simple | Needs careful KV layout in prod |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Represent each job (`DecodeRequest`) |
| 2 | Each step: build tensor from **active only** |
| 3 | Retire on EOS / length |
| 4 | Later: paged KV so memory matches the schedule |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Finished jobs stop; no cross-talk between rows |
| **Soft** | Higher tokens/s on mixed-length traffic |
| **Soft** | Steadier tail latency under load |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Better schedule | More throughput, same weights |
| Coupled to KV layout | Prod needs paging / blocks |
| Ignoring batching | Leaves GPU on the table |

### 6. This course

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## Run

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## Next

**Paged KV · CUDA Graph** (`/paged-cuda-graph/`).
