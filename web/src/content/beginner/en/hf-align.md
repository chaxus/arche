## Idea

**HF align** runs the same short prompt through arche and Hugging Face Llama 3.2, then compares logits—max absolute diff and whether argmax matches. That’s the Part 1 proof your key map, RMSNorm, RoPE, GQA, SwiGLU, and stack layout aren’t merely shape-correct. Fix dtype, load both models, encode one prompt, forward, and report; skip cleanly when weights or login aren’t available.

### What the script reports

[`compare_logits_to_hf`](https://github.com/chaxus/arche/blob/main/arche/align_hf.py):

```python
hf_logits = hf(ids).logits
arche_logits, _ = arche(ids)
max_abs = (arche_logits - hf_logits).abs().max()
argmax_match = torch.equal(arche_logits.argmax(-1), hf_logits.argmax(-1))
```

`run_align_or_explain` returns `None` when HF weights aren’t reachable—**skip ≠ broken lesson logic**.

### When numbers disagree

| Failure mode | Likely cause |
| --- | --- |
| Huge `max_abs` | Wrong map, RoPE theta, norm, or head layout |
| Argmax mismatch | Same—plus dtype / device mismatch |
| Skip | No login / gated / offline—fix env, not formulas |

### Run the check

| Step | Do |
| --- | --- |
| 1 | Login if needed; same model id as weights chapter |
| 2 | float32 CPU compare for stability in the lab |
| 3 | Short prompt (“The cat sat”) |
| 4 | Assert within tolerance or print skip reason |

### Check yourself

| Must hold | With weights: argmax match; `max_abs` under lab tol |
| Must hold | Without weights: explicit skip, not a fake pass |
| Worth aiming for | Longer prompts / more positions in CI later |

### If you fail or pass

| Lever | Impact |
| --- | --- |
| Fail → revisit | Maps / RoPE / norms before “optimizing” |
| Pass | Safe to move to align (SFT/DPO) and serve |
| Engine CI | Same idea gates vLLM-class numeric regs |

### This course

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
