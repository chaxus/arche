## Idea

### 1. Why a block

One reusable unit you can stack `N` times.

### 2. What it is

```
x → RMSNorm → Attn → + residual
  → RMSNorm → SwiGLU → + residual
```

Same shape in/out. Optional KV cache through attention.

### 3. Layout choices

| Choice | Plain idea | Today |
| --- | --- | --- |
| Post-norm | Norm after residual | Older |
| **Pre-norm** | Norm before sublayer | **Modern default** |
| Parallel attn∥MLP | Some non-Llama layouts | Not our trunk |

### 4. What labs ship

Llama 3.2 1B: **16** such blocks.

### 5. What “good” means

- Residual paths keep shapes  
- Cache threading doesn’t drop layers  

### 6. What we do

`TransformerBlock` in `arche/block.py`.

## Run

```bash
python chapters/01-inference/ep08_block.py
```

## Next

**Stack + LM head** (`?c=stack`).
