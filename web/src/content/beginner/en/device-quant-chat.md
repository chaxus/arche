## Idea

### 1. What it is for

Make the model **run and chat**: pick a device, shrink weights, format role messages with a chat template. Not new operators—required glue.

1. **Device** — CUDA → MPS → CPU  
2. **INT8 (teaching)** — smaller weights; dequant then compute  
3. **Chat template** — roles → model string (HF template or text fallback)

Try the page demo.

### 2. Why the LLM pipeline needs it

A correct trunk still fails users without a device, enough RAM, or the right message format. Device = can run; quant = can fit; template = “chat” vs raw completion.

### 3. Where it comes from · how you build it

| Topic | Approach | Production note |
| --- | --- | --- |
| Device | Probe backends; `.to(device)` | Same graph, different backends |
| Quant | Teaching INT8; products use INT4/FP8/… | Accuracy vs speed |
| Template | Model’s own chat template | **Don’t invent** formats |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Fallback chain works | No CUDA shouldn’t hard-crash |
| **Hard** | Template matches model | Wrong template → junk |
| **Soft** | Quality after quant OK | Evals; teaching INT8 ≠ product |
| **Soft** | Clear multi-turn roles | System/user/assistant boundaries |

### 5. What labs ship · what we do

Teaching helpers + script—not a full quant product.

## Run

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## Next

**Continuous batching** (`?c=continuous-batching`).
