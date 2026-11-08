## Idea

**What:** **Autoregressive generation** — turn logits into text by looping: score → pick next id → append → stop.

**Why:** A trained/loaded stack only emits scores. Products differ in sampling and stop rules; they share this loop. Without it you have a classifier-shaped forward, not a chatbot.

**How:** Last-position logits → optional temperature / top-k → greedy or sample → append → halt on EOS or max length. KV cache (next part) makes the same loop fast.

### 1. What — skeleton

[`arche/generate.py`](https://github.com/chaxus/arche/blob/main/arche/generate.py) roughly:

```python
# each step
logits, caches = model(ids_or_last_token, caches=caches, start_pos=...)
next_id = sample(logits[:, -1, :], temperature=..., top_k=...)  # or argmax
# append; break on EOS / max_new_tokens
```

Prefill runs the full prompt once; decode feeds one new token at a time when cache is on.

### 2. Why — knobs ≠ weights

| Knob | Changes |
| --- | --- |
| Temperature / top-k / top-p | Style of picks—not the model floats |
| Greedy argmax | Deterministic, often dull |
| EOS / max length | UX and safety bounds |
| Chat template | How roles become ids (later device/quant/chat) |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Encode prompt with the **paired** tokenizer |
| 2 | Forward → take last time’s logits |
| 3 | Sample or argmax → append id |
| 4 | Repeat until stop; decode ids → text |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Stops; only past tokens visible; valid vocab ids |
| **Soft** | With real weights, coherent short completions |
| **Soft** | Same prompt + greedy ≈ stable across runs |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Sampling only | Behavior without retraining |
| No KV cache | Correct but O(T²) feel—fix next |
| Constrained decoding | Structured outputs |
| Bad stop rules | Runaway or clipped answers |

### 6. This course

```bash
python chapters/01-inference/ep11_generate.py
```

## Run

```bash
python chapters/01-inference/ep11_generate.py
```

## Next

**HF align** (`/hf-align/`): prove the handwritten stack matches HF.
