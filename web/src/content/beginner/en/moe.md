## In one sentence
Mixture-of-Experts runs only the top-k expert MLPs per token—sparse compute, dense parameter count.

## Before you start
- SwiGLU understood. Time: ~15 minutes.

## The idea (plain language)
A **router** scores experts; each token picks `top_k` and blends their outputs.
You can store many experts while each token only pays for a few—ideas behind Mixtral-style stacks.

## In the code
- `arche/moe.py`
- `chapters/01-inference/ep14_moe.py`

## Do this now
```bash
python chapters/01-inference/ep14_moe.py
```

## If something breaks
- `top_k > num_experts` asserts—keep demo defaults.

## Don't worry about yet
Load-balancing losses and production expert parallelism.

## Next
**MLA** (`?c=mla`).
