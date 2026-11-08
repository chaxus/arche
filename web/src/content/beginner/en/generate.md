## In one sentence
Turn logits into tokens: sample (or greedily pick) the next id, append, repeat until stop.

## Before you start
- Weights chapter understood (even if download deferred). Time: ~15 minutes.

## The idea (plain language)
Each step looks at the last position’s scores. Optional **temperature** sharpens/softens; **top-k** keeps only the best k options. Append the chosen id; stop on EOS or length limit.

## In the code
- `arche/generate.py`
- `chapters/01-inference/ep11_generate.py`

## Do this now
```bash
python chapters/01-inference/ep11_generate.py
```
Without weights the tiny demo still shows the loop shape.

## If something breaks
- Empty/gibberish text with random weights is expected—real fluency needs loaded 1B weights.

## Don't worry about yet
KV cache speedups (next part) or chat templates (later).

## Next
Part 2 begins: **KV cache** (`?c=kv-cache`).
