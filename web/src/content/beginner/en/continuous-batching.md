## Idea

**Continuous batching** is the serving **scheduler** shape: many users share decode steps; finished jobs retire and new ones admit mid-flight. A single-user generate loop wastes GPU when one sequence ends early. Engines like vLLM, TGI, and SGLang keep the device busy by stepping only **active** requests together. Active set → pad actives → sample → retire on EOS/max → repeat—the teaching scheduler is not a full paged-attention engine.

### Scheduler loop

[`generate_continuous`](https://github.com/chaxus/arche/blob/main/arche/batching.py):

```python
while any(not item.done for item in live):
    active = [item for item in live if not item.done]
    # pad active rows → model(batch) → sample per row → retire finished
```

Each request carries its own temperature / top-k / max tokens. No weight change—pure scheduling.

### Throughput picture

| Single loop | Continuous batch |
| --- | --- |
| One user owns the step | Many share a step |
| Idle after short finish | Retire + admit keeps GPU full |
| Simple | Needs careful KV layout in prod |

### Build the toy scheduler

| Step | Do |
| --- | --- |
| 1 | Represent each job (`DecodeRequest`) |
| 2 | Each step: build tensor from **active only** |
| 3 | Retire on EOS / length |
| 4 | Later: paged KV so memory matches the schedule |

### Check yourself

| Must hold | Finished jobs stop; no cross-talk between rows |
| Worth aiming for | Higher tokens/s on mixed-length traffic |
| Worth aiming for | Steadier tail latency under load |

### Serving ceiling

| Lever | Impact |
| --- | --- |
| Better schedule | More throughput, same weights |
| Coupled to KV layout | Prod needs paging / blocks |
| Ignoring batching | Leaves GPU on the table |

### This course

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## Run

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## Next

**Paged KV · CUDA Graph** (`/paged-cuda-graph/`).
