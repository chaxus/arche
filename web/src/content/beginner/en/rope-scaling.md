## In one sentence
Llama 3.2 stretches long wavelengths so context can grow far past the original window.

## Before you start
- RoPE + KV cache ideas clear. Time: ~15 minutes.

## The idea (plain language)
A plain RoPE trained for shorter context misbehaves if you suddenly ask for 100k tokens.
Llama 3.2 keeps a large `rope_theta` and applies a **factor / low / high** schedule: short wavelengths stay, long ones stretch, middle band smooths.

## In the code
- Scaling lives in `arche/rope.py` / attention config
- `chapters/01-inference/ep13_rope_scaling.py`

## Do this now
```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## If something breaks
- Tiny demos may not show 128k—look at printed frequency changes instead.

## Don't worry about yet
YaRN vs NTK debates; we teach the Llama 3.2 schedule used here.

## Next
**MoE** (`?c=moe`).
