## Idea

**Autoregressive generation** is the loop that turns logits into text: score → pick the next token id → append → stop. A trained or loaded stack only emits scores until you wrap it in this loop; products differ in sampling and stop rules, but they all share the same shape. Take last-position logits, optional temperature / top-k, greedy or sample, halt on EOS or max length—KV cache (next chapter) makes the same loop fast at long outputs.

### Loop skeleton

[`arche/generate.py`](https://github.com/chaxus/arche/blob/main/arche/generate.py) roughly:

```python
# each step
logits, caches = model(ids_or_last_token, caches=caches, start_pos=...)
next_id = sample(logits[:, -1, :], temperature=..., top_k=...)  # or argmax
# append; break on EOS / max_new_tokens
```

Prefill runs the full prompt once; decode feeds one new token at a time when cache is on.

### Knobs are not weights

| Knob | Changes |
| --- | --- |
| Temperature / top-k / top-p | Style of picks—not the model floats |
| Greedy argmax | Deterministic, often dull |
| EOS / max length | UX and safety bounds |
| Chat template | How roles become ids (later device/quant/chat) |

### Wire it up

| Step | Do |
| --- | --- |
| 1 | Encode prompt with the **paired** tokenizer |
| 2 | Forward → take last time’s logits |
| 3 | Sample or argmax → append id |
| 4 | Repeat until stop; decode ids → text |

### Check yourself

| Must hold | Stops; only past tokens visible; valid vocab ids |
| Worth aiming for | With real weights, coherent short completions |
| Worth aiming for | Same prompt + greedy ≈ stable across runs |

### Tuning behavior without retraining

| Lever | Impact |
| --- | --- |
| Sampling only | Behavior without retraining |
| No KV cache | Correct but O(T²) feel—fix next |
| Constrained decoding | Structured outputs |
| Bad stop rules | Runaway or clipped answers |

### This course

```bash
python chapters/01-inference/ep11_generate.py
```

## Run

```bash
python chapters/01-inference/ep11_generate.py
```

## Next

**HF align** (`/hf-align/`): prove the handwritten stack matches HF.
