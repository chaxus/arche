## Idea

**What:** **Mixture-of-Experts (MoE)** — replace the dense SwiGLU with **many experts + a router**; each token runs **top-k** only.

**Why:** Huge parameter capacity at near-dense **per-token** compute. **DeepSeek-V3**’s open story is largely **MoE (capacity) + MLA (KV)**; Mixtral is another open MoE. This course’s Llama 3.2 example stayed **dense** on purpose—MoE is the map after you can read the dense FFN.

**How:** Pick `E` / `top_k`; experts + gate; aux load-balance in real trains; sparse infer; shard experts in checkpoints.

### 1. What — forward

[`MoESwiGLU`](https://github.com/chaxus/arche/blob/main/arche/moe.py):

```python
logits = self.gate(x)
probs = F.softmax(logits, dim=-1)
top_probs, top_idx = probs.topk(self.top_k, dim=-1)
# weighted sum of selected SwiGLU experts
```

Same decoder spine; only the MLP slot changes.

### 2. Why — dense vs sparse

| | Dense SwiGLU | MoE |
| --- | --- | --- |
| Params | One FFN | `E` × FFN |
| Per-token FLOPs | Full FFN | ≈ `top_k` experts |
| Failure mode | — | Expert collapse without balancing |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Keep attention / norms; swap MLP → MoE |
| 2 | Choose `E`, `top_k` (e.g. Mixtral-like top-2) |
| 3 | Add load-balance loss in serious training |
| 4 | Plan expert-parallel / sharded loads |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Valid routes; merge weights sum sanely |
| **Soft** | Balanced expert load; quality/capacity trade OK |
| **Soft** | Checkpoint maps cover all experts |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| More `E` / `k` | Capacity vs compute |
| Balance loss | Fights collapse |
| Expert parallel | Multi-GPU engineering cost |

### 6. Industry map

| Line | Role |
| --- | --- |
| **DeepSeek-V3** | Large MoE + MLA (next) |
| **Mixtral** | Top-2 SwiGLU experts |
| **Llama 3.2 ~1B (course)** | Dense first |

### 7. This course

```bash
python chapters/01-inference/ep14_moe.py
```

## Run

```bash
python chapters/01-inference/ep14_moe.py
```

## Next

**MLA** (`/mla/`): compress KV the DeepSeek way.
