## Idea

**What:** **SwiGLU** — gated feed-forward: `down(silu(gate(x)) * up(x))` with three **bias-free** linears.

**Why:** Attention mixes *across* positions; the FFN transforms *each* position. SwiGLU (not plain GELU MLP) is the dense default in Llama/Qwen; MoE experts are often the same unit.

**How:** Pick `intermediate_dim`, three mats, wire after the second pre-norm in the block.

### 1. What — code

[`arche/mlp.py`](https://github.com/chaxus/arche/blob/main/arche/mlp.py):

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

Gate and up expand `D → I`; down returns `I → D`. No bias—matches Llama weight files.

### 2. Why — gated vs plain MLP

| Choice | Why |
| --- | --- |
| SiLU gate × up | Soft gating; empirically strong for LMs |
| Three mats | Slightly more params than two-mat GELU MLP at same `I` |
| No bias | Smaller, matches open Llama/Qwen tensors |
| Position-wise | Parallelizable; pairs with residual |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Choose `I` (Llama 3.2 1B-style intermediate from config) |
| 2 | `gate_proj`, `up_proj`, `down_proj` bias=False |
| 3 | Place after `post_attention_layernorm` |
| 4 | MoE later: many SwiGLUs + router |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | In/out dim `D`; no bias; activation is SiLU on gate |
| **Soft** | Judge via full-model loss / quality, not isolated FFN |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Wider `I` | Capacity and FLOPs |
| GELU → SwiGLU | Needs retrain |
| MoE experts | Same brick × E, sparse fire |

### 6. This course

```bash
python chapters/01-inference/ep07_swiglu.py
```

## Run

```bash
python chapters/01-inference/ep07_swiglu.py
```

## Next

**Transformer block** (`/block/`): sandwich the pieces.
