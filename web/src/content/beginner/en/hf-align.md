## Idea

### 1. What it is for

**HF align**: same short prompt through arche and official HF Llama 3.2; compare argmax and max abs logit diff—evidence the dense trunk and weight map are trustworthy.

### 2. Why the LLM pipeline needs it

Hand-written stacks can “run” with wrong layers. Matching a reference on short logits is hard proof. No weights → clean skip ≠ broken logic.

### 3. Where it comes from · how you build it

This script + CI synthetic map tests; license/login for real weights.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Logits close enough with weights | Map/formulas OK |
| **Hard** | Skip clearly without weights | Don’t fake pass |
| **Soft** | Longer seqs / more ops | Later; short prompt here |

### 5. What we do

See Run; Fix if gated.

## Run

```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
```

## Fix

No weights / not logged in → skip or tip; not a logic failure.

## Next

**Sharded MoE/MLA** (`?c=sharded-frontier`).
