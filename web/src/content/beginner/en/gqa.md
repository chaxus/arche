## Idea

### 1. What it is for

**GQA** lets groups of Q heads **share** fewer KV heads. Still causal (`t` sees only `≤ t`), but decode stores less K/V.

Flow: project → split heads → RoPE → expand KV → SDPA → out proj. See the page viz.

### 2. Why the LLM pipeline needs it

Full MHA: every Q head owns KV → **KV cache** grows fast. GQA trades a bit of flexibility for much smaller cache vs MHA; usually gentler than extreme MQA.

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Spec** | Choose `n_q`, `n_kv` (divides evenly); `n_rep = n_q / n_kv` |
| **Impl** | Broadcast/repeat KV per group, then attend |
| **Train** | Learn projections with the net |
| **Infer** | Same layout + cache for prefill/decode |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | `n_q` divisible by `n_kv` | Valid groups |
| **Hard** | Causal mask correct | No future tokens |
| **Hard** | Head counts match weights | Else shape/silent errors |
| **Soft** | Cache clearly smaller than MHA | The point of GQA |
| **Soft** | Quality doesn’t collapse vs MHA | Judge with evals/loss |

### 5. What labs ship · what we do

Llama 3.2 1B: **32 Q / 8 KV**.  
`CausalSelfAttention` in `arche/attention.py`; lab prints heads and shapes.

## Run

```bash
python chapters/01-inference/ep06_gqa.py
```

## Next

**SwiGLU** (`?c=swiglu`): the feed-forward half.
