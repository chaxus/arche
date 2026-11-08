## Idea

The **dense causal LM shell** wires embedding, `Block×N`, final RMSNorm, and the vocabulary head: `ids → embed → Block×N → final RMSNorm → Linear(V) → logits`. A single block is not yet a language model—this stack is what pretrain’s `model(batch)` has been calling from the outside. Build from config (`V,D,N,heads,…`), optionally **tie** LM head and embeddings (common on small Llamas), then init, load, or pretrain.

### Forward pass

[`Llama`](https://github.com/chaxus/arche/blob/main/arche/model.py):

```python
x = self.embed_tokens(input_ids)
for layer, cache in zip(self.layers, caches):
    x, new_cache = layer(x, cache=cache, start_pos=start_pos)
x = self.norm(x)
return self.lm_head(x), new_caches
```

Logits shape `(B, T, V)` — exactly what `lm_loss` consumes after shift.

### Tying and final norm

| Choice | Why it matters |
| --- | --- |
| Final RMSNorm | Stabilize before vocab projection |
| Tie head↔embed | Fewer params; common at ~1B |
| Untied head | More freedom at larger scales |
| Shared config object | One source of truth for HF map |

### Build from scratch

| Step | Do |
| --- | --- |
| 1 | Fill `Llama32Config` (vocab, dims, heads, rope, …) |
| 2 | `ModuleList` of `N` blocks |
| 3 | Set `tie_word_embeddings` before init/load |
| 4 | Train with pretrain loop **or** `from_pretrained` |

### Check yourself

| Must hold | Logits `(B,T,V)`; config ↔ weights; tying consistent |
| Worth aiming for | Quality after load or enough pretrain steps |

### Tuning the whole model

| Lever | Impact |
| --- | --- |
| Depth / width / V | Architecture-level; retrain or migrate carefully |
| MoE swap | Replaces SwiGLU inside blocks—skeleton stays |
| Optimize modules | Don’t “tune the stack alone” without data/optim |

### This course

```bash
python chapters/01-inference/ep09_stack.py
```

## Run

```bash
python chapters/01-inference/ep09_stack.py
```

## Next

**Load weights** (`/weights/`): fill the shell with Llama floats.
