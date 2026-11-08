## Idea

Two serving accelerators share this lesson: **paged KV** (block table + shared page pool) fights memory fragmentation, and **CUDA Graph** captures a fixed-shape decode step to cut launch overhead. Continuous batching without paging wastes VRAM on ragged lengths; eager CUDA launches tax every tiny decode. Together they point at vLLM-class engines—here you get a teaching stand-in, not a drop-in replacement. Logical KV → alloc/free pages via block table → `gather_kv` for attention; optionally capture graphs when decode shapes stay stable.

### Paged KV

[`PagedKVPool` / `BlockTable` / `gather_kv`](https://github.com/chaxus/arche/blob/main/arche/paged_kv.py):

```python
# append one step into the next slot; alloc a new page when a page fills
append_kv(pool, table, k, v)
# materialize contiguous KVCache for SDPA
cache = gather_kv(pool, table)
```

Physical pages are shared; each sequence owns a list of page ids.

### CUDA Graph (optional)

[`capture_module_step` / `generate_with_optional_graph`](https://github.com/chaxus/arche/blob/main/arche/cuda_graphs.py): capture a module step when CUDA is available; replay instead of re-launching. Needs **stable shapes**—why engines care about batching policy.

### Toward real engines

| Piece | Pain it hits |
| --- | --- |
| Paging | Fragmentation + multi-user packing |
| Graphs | Per-step CPU launch overhead |
| Neither changes weights | Raises the serving ceiling only |

### Wire it up

| Step | Do |
| --- | --- |
| 1 | Continuous batch mental model first |
| 2 | Pool + per-seq block table |
| 3 | Gather before attention (teaching path) |
| 4 | Optional graph on decode when shapes freeze |

### Check yourself

| Must hold | Gather matches logical cache; tables in-bounds |
| Worth aiming for | Less fragmentation / launch tax; higher throughput |
| Worth aiming for | Clear skip when no CUDA |

### This course

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

Part 3 ends; Part 4 maps MoE/MLA frontier structures.

## Run

```bash
python chapters/01-inference/ep23_paged_cuda_graph.py
```

## Next

**MoE** (`/moe/`): frontier capacity via sparse experts.
