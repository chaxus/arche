## Idea

### 1. What it is for

A **Transformer block** is one repeatable sandwich: RMSNorm → attention → residual → RMSNorm → SwiGLU → residual. Same shape in/out so you can stack `N` times.

### 2. Why the LLM pipeline needs it

Parts alone aren’t a deep LM. The block is the unit of depth: residuals keep a path for the original signal; pre-norm keeps scales sane at each entry.

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Wire** | Sandwich above; attention may take a KV cache |
| **Depth** | Llama 3.2 1B: **16** blocks |
| **Train** | Each block has its own weights |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Residual shapes match | Can add |
| **Hard** | Pre-norm placement | Llama contract |
| **Hard** | Per-layer cache if used | No cross-talk / missing layers |
| **Soft** | Still trainable when deep | Residuals + norms |

### 5. What labs ship · what we do

Llama-style pre-norm blocks.  
`TransformerBlock` in `arche/block.py`.

## Run

```bash
python chapters/01-inference/ep08_block.py
```

## Next

**Stack + LM head** (`?c=stack`).
