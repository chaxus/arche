## Idea

**What:** The full **dense causal LM shell**: `ids → embed → Block×N → final RMSNorm → Linear(V) → logits`.

**Why:** One block is not a language model. The stack is what `model(batch)` in pretrain called—a black box until now.

**How:** Build from config (`V,D,N,heads,…`); optional **tie** LM head ↔ embed (small Llamas often do); init, then load or pretrain.

### 1. What — forward

[`Llama`](https://github.com/chaxus/arche/blob/main/arche/model.py):

```python
x = self.embed_tokens(input_ids)
for layer, cache in zip(self.layers, caches):
    x, new_cache = layer(x, cache=cache, start_pos=start_pos)
x = self.norm(x)
return self.lm_head(x), new_caches
```

Logits shape `(B, T, V)` — exactly what `lm_loss` consumes after shift.

### 2. Why — tying and final norm

| Choice | Why |
| --- | --- |
| Final RMSNorm | Stabilize before vocab projection |
| Tie head↔embed | Fewer params; common at ~1B |
| Untied head | More freedom at larger scales |
| Shared config object | One source of truth for HF map |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Fill `Llama32Config` (vocab, dims, heads, rope, …) |
| 2 | `ModuleList` of `N` blocks |
| 3 | Set `tie_word_embeddings` before init/load |
| 4 | Train with pretrain loop **or** `from_pretrained` |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Logits `(B,T,V)`; config ↔ weights; tying consistent |
| **Soft** | Quality after load or enough pretrain steps |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Depth / width / V | Architecture-level; retrain or migrate carefully |
| MoE swap | Replaces SwiGLU inside blocks—skeleton stays |
| Optimize modules | Don’t “tune the stack alone” without data/optim |

### 6. This course

```bash
python chapters/01-inference/ep09_stack.py
```

## Run

```bash
python chapters/01-inference/ep09_stack.py
```

## Next

**Load weights** (`/weights/`): fill the shell with Llama floats.
