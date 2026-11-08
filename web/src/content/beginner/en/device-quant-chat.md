## Idea

### 1. Why these three

Math core isn’t enough to *run* a chat: device, size, and message format matter.

### 2. What we cover

1. **Device** — CUDA → MPS → CPU  
2. **INT8 (teaching)** — smaller weights; simple dequant path  
3. **Chat template** — roles → string the model expects (HF or plain fallback; try the demo)

### 3. Landscape

| Topic | Production note |
| --- | --- |
| Quant | INT8/INT4/FP8… many recipes |
| Templates | Model-specific; don’t invent casually |
| Devices | Same graph, different backends |

### 4–6. What we do

Teaching helpers + chapter script—not a full quant product.

## Run

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## Next

**Continuous batching** (`?c=continuous-batching`).
