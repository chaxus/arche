## Idea

### 1. What it is for

**RoPE scaling** adjusts rotations when context exceeds the training length—Llama 3 style: large `rope_theta` plus factor/low/high **wavelength smoothing**.

### 2. Why the LLM pipeline needs it

RoPE angles were tuned for the train length. Naively huge positions distort relative distance. Products wanting 128K-class context often add a scaling schedule (and/or train longer) without inventing a new position system.

### 3. Where it comes from · how you build it

| Path | Plain meaning |
| --- | --- |
| **Llama 3 wavelength smooth** | This lesson; HF-compatible |
| NTK / YaRN / … | Other long-context recipes |
| Train longer | Not formula-only |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Matches checkpoint scaling config | Wrong recipe → long-context junk |
| **Soft** | Near distances still familiar | Don’t break short deps |
| **Soft** | Far distances usable | Long-context evals |

### 5. What labs ship · what we do

Llama 3.2 official long-context schedule.  
Scaling helpers in `arche/rope.py` + this script.

## Run

```bash
python chapters/01-inference/ep13_rope_scaling.py
```

## Next

**MoE** (`?c=moe`).
