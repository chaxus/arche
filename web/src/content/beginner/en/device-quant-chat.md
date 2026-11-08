## Idea

Three practical pieces that sit around the math core:

1. **Device** — pick CUDA if present, else Apple MPS, else CPU (`pick_device`)  
2. **INT8 (teaching)** — store weights in int8 to shrink memory; compute still follows a simple dequant path for learning  
3. **Chat template** — turn role messages (system/user/assistant) into the token string the model expects; if HF’s template is missing, we fall back to a plain readable format (try the demo on this page)

## Run

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

Expect device choice, a quant note, and a rendered chat string.

## Next

**Continuous batching** (`?c=continuous-batching`): many short generations sharing steps.
