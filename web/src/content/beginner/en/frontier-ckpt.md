## In one sentence
Save/load MoE and MLA modules and map Mixtral-style expert keys into arche.

## Before you start
- MoE + MLA primers. Time: ~15 minutes.

## The idea (plain language)
Frontier pieces need checkpoints too. We round-trip state dicts and translate Mixtral `w1/w2/w3` names into our SwiGLU experts.
Sharded indexes continue in the next chapter.

## In the code
- `arche/checkpoint.py`, `load_mixtral.py`
- `chapters/02-training/ep20_frontier_ckpt.py`

## Do this now
```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## If something breaks
- Missing expert keys → ensure the synthetic HF-like dict includes gate + all experts.

## Don't worry about yet
Downloading full Mixtral 8x7B.

## Next
**HF logits alignment** (`?c=hf-align`).
