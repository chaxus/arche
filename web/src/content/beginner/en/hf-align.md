## In one sentence
Prove the hand-written graph matches Hugging Face Llama 3.2 logits on a short prompt when weights are available.

## Before you start
- Weights + generate chapters. Time: ~20 minutes.
- Needs gated weights / login.

## The idea (plain language)
Same weights + same math should agree on argmax and stay within a small max-abs drift.
If weights are offline, the chapter exits cleanly; CI still checks synthetic bit-exact mapping.

## In the code
- `arche/align_hf.py`
- `chapters/01-inference/ep21_hf_align.py`
- `tests/test_align_hf.py`

## Do this now
```bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
# or: ARCH_HF_ALIGN=1 pytest tests/test_align_hf.py -q
```

## If something breaks
- Skipped/unreachable → finish HF setup; synthetic test still validates the mapper.

## Don't worry about yet
Bitwise identity across every kernel and dtype.

## Next
**Sharded MoE / MLA** (`?c=sharded-frontier`).
