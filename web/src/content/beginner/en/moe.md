## Idea

**Mixture-of-Experts (MoE)** swaps the dense SwiGLU for many experts plus a router; each token only runs **top-k** of them. You get huge parameter capacity while per-token compute stays near the dense case. DeepSeek-V3’s open story is largely MoE for capacity and MLA for KV; Mixtral is another open MoE. This course stayed on dense Llama 3.2 first so you can read the FFN before the sparse map.

Pick `E` and `top_k`, wire experts and a gate, add load-balance loss in serious training, and plan how experts land in checkpoints.

### Forward pass

[`MoESwiGLU`](https://github.com/chaxus/arche/blob/main/arche/moe.py):

```python
logits = self.gate(x)
probs = F.softmax(logits, dim=-1)
top_probs, top_idx = probs.topk(self.top_k, dim=-1)
# weighted sum of selected SwiGLU experts
```

The decoder spine stays the same; only the MLP slot changes.

### Dense vs sparse

| | Dense SwiGLU | MoE |
| --- | --- | --- |
| Params | One FFN | `E` × FFN |
| Per-token FLOPs | Full FFN | ≈ `top_k` experts |
| Failure mode | — | Expert collapse without balancing |

### From zero

| Step | Do |
| --- | --- |
| 1 | Keep attention / norms; swap MLP → MoE |
| 2 | Choose `E`, `top_k` (e.g. Mixtral-like top-2) |
| 3 | Add load-balance loss in serious training |
| 4 | Plan expert-parallel / sharded loads |

### Check yourself

- Routes are valid; merge weights sum sanely.
- Expert load stays roughly balanced; quality/capacity trade feels OK.
- Checkpoint maps cover every expert.

### Levers

| Lever | Impact |
| --- | --- |
| More `E` / `k` | Capacity vs compute |
| Balance loss | Fights collapse |
| Expert parallel | Multi-GPU engineering cost |

### Industry map

| Line | Role |
| --- | --- |
| **DeepSeek-V3** | Large MoE + MLA (next) |
| **Mixtral** | Top-2 SwiGLU experts |
| **Llama 3.2 ~1B (course)** | Dense first |

### This course

```bash
python chapters/01-inference/ep14_moe.py
```

## Run

```bash
python chapters/01-inference/ep14_moe.py
```

## Next

**MLA** (`/mla/`): compress KV the DeepSeek way.
