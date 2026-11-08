## Idea

This chapter is shipping glue: **device** placement, a teaching **INT8** shrink, and **chat templates** that turn roles into token ids. A correct dense stack still fails users if it won’t run on their hardware, doesn’t fit memory, or formats chat wrong—these aren’t new layers, they’re the last mile. Probe device, optional quant recipe, apply **this model’s** template, then wire into generate and KV.

### Pieces in the repo

| Piece | Role in repo |
| --- | --- |
| Device fallback | CPU/CUDA/MPS pick for demos |
| INT8 toy | Teaching shrink—not GPTQ/AWQ/FP8 production |
| Chat template | [`apply_chat_template` / `encode_chat`](https://github.com/chaxus/arche/blob/main/arche/chat.py) |

Wrong template ⇒ model “ignores” system/user structure even with perfect weights.

### Product constraints

| Pressure | Response |
| --- | --- |
| No GPU | CPU path must work (slow OK for labs) |
| VRAM tight | Quant / smaller dtype |
| Multi-turn UX | Official chat template + specials |
| Multi-backend | Engineering cost, not math |

### Ship a demo

| Step | Do |
| --- | --- |
| 1 | Probe available device |
| 2 | Load weights in a runnable dtype |
| 3 | Format messages with the **paired** template |
| 4 | Generate with KV as usual |

### Check yourself

| Must hold | Fallback runs; template matches tokenizer/model |
| Worth aiming for | Quant quality acceptable for the demo bar |
| Worth aiming for | Roles render clearly in the prompt string/ids |

### Beyond the toy path

| Lever | Impact |
| --- | --- |
| Harder quant | Smaller/faster/riskier |
| Wrong template | Junk dialogues |
| Prod quant (GPTQ/AWQ/FP8) | Engine territory (vLLM/TGI/llama.cpp) |

### This course

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## Run

```bash
python chapters/01-inference/ep16_device_quant_chat.py
```

## Next

**Continuous batching** (`/continuous-batching/`).
