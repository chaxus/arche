## In one sentence
RoPE rotates query/key vectors so attention “knows” relative positions—no absolute position table required.

## Before you start
- RMSNorm done. Time: ~15 minutes.

## The idea (plain language)
Instead of adding a position vector, **RoPE** rotates pairs of dimensions by an angle that depends on token index. Relative distance falls out of the dot product.
Llama 3.2 also uses a large `rope_theta` and later a long-context **scale**—that scale is its own chapter.

## Picture / formula
$$q_m' = R_{\Theta,m} q_m,\quad k_n' = R_{\Theta,n} k_n$$

## In the code
- `arche/rope.py`
- `chapters/01-inference/ep05_rope.py`

## Do this now
```bash
python chapters/01-inference/ep05_rope.py
```

## If something breaks
- Odd head dimensions break pair rotation—use the script’s defaults.

## Don't worry about yet
Every RoPE paper variant; long-context scaling waits for ep13.

## Next
**GQA attention** (`?c=gqa`).
