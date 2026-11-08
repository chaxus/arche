## Idea

**What:** Shipping glue — **device** placement, a teaching **INT8** shrink, and **chat templates** that turn roles into token ids.

**Why:** A correct dense stack still fails users if it can’t run on their box, doesn’t fit memory, or formats chat wrong. These are not new layers; they are the last mile.

**How:** Probe device → optional quant recipe → apply **this model’s** template → wire into generate / KV.

### 1. What — pieces

| Piece | Role in repo |
| --- | --- |
| Device fallback | CPU/CUDA/MPS pick for demos |
| INT8 toy | Teaching shrink—not GPTQ/AWQ/FP8 production |
| Chat template | [`apply_chat_template` / `encode_chat`](https://github.com/chaxus/arche/blob/main/arche/chat.py) |

Wrong template ⇒ model “ignores” system/user structure even with perfect weights.

### 2. Why — product constraints

| Pressure | Response |
| --- | --- |
| No GPU | CPU path must work (slow OK for labs) |
| VRAM tight | Quant / smaller dtype |
| Multi-turn UX | Official chat template + specials |
| Multi-backend | Engineering cost, not math |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Probe available device |
| 2 | Load weights in a runnable dtype |
| 3 | Format messages with the **paired** template |
| 4 | Generate with KV as usual |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Fallback runs; template matches tokenizer/model |
| **Soft** | Quant quality acceptable for the demo bar |
| **Soft** | Roles render clearly in the prompt string/ids |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Harder quant | Smaller/faster/riskier |
| Wrong template | Junk dialogues |
| Prod quant (GPTQ/AWQ/FP8) | Engine territory (vLLM/TGI/llama.cpp) |

### 6. This course

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## Run

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## Next

**Continuous batching** (`/continuous-batching/`).
