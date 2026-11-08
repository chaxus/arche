## Idea

From-zero models must **ship to humans**: device, size, chat format—glue, not new layers.

### 1. What it is for

Device fallback; teaching INT8 shrink; chat templates.

### 2. From zero: how you build it

Probe device → pick quant recipe (INT8 toy; GPTQ/AWQ/FP8 in prod) → **model’s own** template → wire with generate/KV.

### 3. What “good” means

Fallback works; template paired; quant quality OK; clear roles.

### 4. Improve · whole-model impact

Harder quant ⇒ smaller/faster/riskier; wrong template ⇒ junk; multi-backend is engineering cost.

### 5. Industry · open models

HF chat templates; vLLM/TGI load quant+template; Llama/Qwen ship official templates; llama.cpp/GGML quants.

### 6. This course

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## Run

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## Next

**Continuous batching** (`?c=continuous-batching`).
