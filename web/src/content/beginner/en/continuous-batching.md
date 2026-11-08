## In one sentence
A teaching scheduler steps many short decode jobs together until each finishes.

## Before you start
- Generation + KV ideas. Time: ~15 minutes.

## The idea (plain language)
Production servers do not run one request after another only. They keep a pool of active sequences, step them together, and retire finished ones.
Our version pads active rows each step—**not** paged attention, but the control-loop shape.

## In the code
- `arche/batching.py`
- `chapters/01-inference/ep17_continuous_batching.py`

## Do this now
```bash
python chapters/01-inference/ep17_continuous_batching.py
```
Use the site demo to watch fake jobs share steps.

## If something breaks
- Empty outputs → ensure `max_new_tokens >= 1`.

## Don't worry about yet
CUDA graphs + paged fusion (ep23 sketches the pieces).

## Next
Part 3: **Pretrain** (`?c=pretrain`).
