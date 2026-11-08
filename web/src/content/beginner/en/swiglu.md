## Idea

**SwiGLU** is the gated feed-forward block: `down(silu(gate(x)) * up(x))` with three **bias-free** linear layers. Attention mixes *across* positions; the FFN transforms *each* position on its own. SwiGLU—not a plain GELU MLP—is the dense default in Llama and Qwen; MoE experts are often the same brick. Pick `intermediate_dim`, three projection matrices, and place the module after the second pre-norm in each block.

### Code in the repo

[`arche/mlp.py`](https://github.com/chaxus/arche/blob/main/arche/mlp.py):

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

Gate and up expand `D → I`; down returns `I → D`. No bias—matches Llama weight files.

### Gated FFN vs plain MLP

| Choice | Rationale |
| --- | --- |
| SiLU gate × up | Soft gating; empirically strong for LMs |
| Three mats | Slightly more params than two-mat GELU MLP at same `I` |
| No bias | Smaller, matches open Llama/Qwen tensors |
| Position-wise | Parallelizable; pairs with residual |

### Wiring from scratch

| Step | Do |
| --- | --- |
| 1 | Choose `I` (Llama 3.2 1B-style intermediate from config) |
| 2 | `gate_proj`, `up_proj`, `down_proj` bias=False |
| 3 | Place after `post_attention_layernorm` |
| 4 | MoE later: many SwiGLUs + router |

### Check yourself

- In/out dim `D`; no bias; SiLU on the gate branch.
- Judge via full-model loss / quality, not an isolated FFN.

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| Wider `I` | Capacity and FLOPs |
| GELU → SwiGLU | Needs retrain |
| MoE experts | Same brick × E, sparse fire |

### This course

```bash
python chapters/01-inference/ep07_swiglu.py
```

## Run

```bash
python chapters/01-inference/ep07_swiglu.py
```

## Next

**Transformer block** (`/block/`): sandwich the pieces.
