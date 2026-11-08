## Idea

### 1. What it is for

**Generation** loops “one forward → next id”: last-position logits → optional temperature / top-k → greedy or sample → append → stop on EOS or length.

### 2. Why the LLM pipeline needs it

One forward only yields the **next-token** distribution, not a full reply. Chat, completion, tools all sit on this loop; products differ in stopping, sampling, and templates.

### 3. Where it comes from · how you build it

| Step | What happens |
| --- | --- |
| **1** | Prefill / forward → logits |
| **2** | Last position; optional temperature, top-k |
| **3** | argmax or sample a new id |
| **4** | Append; stop on EOS or max length |

KV cache (next) makes step 1 cheap after prefill; the **loop** stays the same.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Stops cleanly | EOS / max length work |
| **Hard** | Only past prefix | No future tokens |
| **Soft** | Temperature / top-k behave | 0 greedier; higher more random |
| **Soft** | Coherent text with real weights | Full model + sampling |

### 5. What labs ship · what we do

Every chat stack uses this loop shape.  
`generate` in `arche/generate.py`.

## Run

```bash
python chapters/01-inference/ep11_generate.py
```

## Next

**KV cache** (`?c=kv-cache`).
