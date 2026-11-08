## Idea

**What:** **Numeric check** — same short prompt through arche vs Hugging Face Llama 3.2; compare logits (max abs diff + argmax match).

**Why:** End of Part 1 proof: key map, RMSNorm, RoPE, GQA, SwiGLU, stack layout aren’t “shape-right but wrong.” Catch bugs before serve/optim chapters hide them.

**How:** Fixed dtype; load both models; encode one prompt; forward; report. Skip cleanly if weights/login unavailable.

### 1. What — report

[`compare_logits_to_hf`](https://github.com/chaxus/arche/blob/main/arche/align_hf.py):

```python
hf_logits = hf(ids).logits
arche_logits, _ = arche(ids)
max_abs = (arche_logits - hf_logits).abs().max()
argmax_match = torch.equal(arche_logits.argmax(-1), hf_logits.argmax(-1))
```

`run_align_or_explain` returns `None` when HF weights aren’t reachable—**skip ≠ broken lesson logic**.

### 2. Why — this gate

| Failure mode | Likely cause |
| --- | --- |
| Huge `max_abs` | Wrong map, RoPE theta, norm, or head layout |
| Argmax mismatch | Same—plus dtype / device mismatch |
| Skip | No login / gated / offline—fix env, not formulas |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Login if needed; same model id as weights chapter |
| 2 | float32 CPU compare for stability in the lab |
| 3 | Short prompt (“The cat sat”) |
| 4 | Assert within tolerance or print skip reason |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | With weights: argmax match; `max_abs` under lab tol |
| **Hard** | Without weights: explicit skip, not a fake pass |
| **Soft** | Longer prompts / more positions in CI later |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Fail → revisit | Maps / RoPE / norms before “optimizing” |
| Pass | Safe to move to align (SFT/DPO) and serve |
| Engine CI | Same idea gates vLLM-class numeric regs |

### 6. This course

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

Part 1 closes here. Part 2 steers the base next-token skill (SFT/DPO).

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## Fix

No weights/login → skip means “can’t download,” not “your math is wrong.”

## Next

**SFT · DPO** (`/sft-dpo/`): align after the base next-token skill.
