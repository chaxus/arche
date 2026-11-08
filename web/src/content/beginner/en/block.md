## In one sentence
One transformer block is a pre-norm residual sandwich you can stack.

## Before you start
- Attention + SwiGLU done. Time: ~12 minutes.

## The idea (plain language)
```
x → RMSNorm → Attn → +x → RMSNorm → SwiGLU → +x
```
Same shape in, same shape out—so depth is a loop. Llama 3.2 1B stacks this 16 times (3B: 28).

## In the code
- `arche/block.py`
- `chapters/01-inference/ep08_block.py`

## Do this now
```bash
python chapters/01-inference/ep08_block.py
```

## If something breaks
- Residual add requires matching shapes—don’t change only one branch in experiments.

## Don't worry about yet
Parallel attention/MLP layouts in some giant models.

## Next
**Stack + LM head** (`?c=stack`).
