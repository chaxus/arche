## In one sentence
Pick a device, optionally store weights as INT8, and render chat turns with a template.

## Before you start
- Comfortable running scripts. Time: ~15 minutes.

## The idea (plain language)
`pick_device` prefers CUDA → MPS → CPU.
**INT8 weight-only** keeps weights small and dequantizes for compute (teaching grade, not bitsandbytes).
**Chat templates** format role/content; HF template if present, else a plain fallback.

## In the code
- `arche/device.py`, `quantize.py`, `chat.py`
- `chapters/01-inference/ep16_device_quant_chat.py`

## Do this now
```bash
python chapters/01-inference/ep16_device_quant_chat.py
```
Try the on-page demo too.

## If something breaks
- MPS/CUDA errors → force CPU: the helper accepts `"cpu"`.

## Don't worry about yet
GPTQ/AWQ production quant stacks.

## Next
**Continuous batching** (`?c=continuous-batching`).
