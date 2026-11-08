## Idea

RoPE angles were tuned for a training context length. Push much longer and relative positions can “wrap” oddly.

**Llama 3.2-style RoPE scaling** keeps a large `rope_theta` and smoothly rescales **long wavelengths** using factor / low / high frequency bands (Hugging Face–compatible formula). Short-range behavior stays familiar; long-range gets stretched so longer prompts remain usable.

This chapter is about the **schedule**, not inventing a new attention.

## Run

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

Expect prints that contrast unscaled vs scaled inverse frequencies / shapes.

## Next

**MoE** (`?c=moe`): sparse experts instead of one big MLP.
