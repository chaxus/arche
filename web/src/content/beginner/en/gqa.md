## Idea

### 1. Why GQA

Full MHA: every query head has its own KV head → **KV cache** grows fast at decode.

### 2. What GQA does

**Grouped-Query Attention**: fewer KV heads than query heads; several Q heads **share** one KV. Still **causal** (position `t` sees `≤ t`).

Flow: project → reshape → RoPE → expand KV → SDPA → out proj. See the diagram on this page.

### 3. Family of attention layouts

| Layout | Plain idea | Today |
| --- | --- | --- |
| MHA | Q heads = KV heads | Classic |
| MQA | One KV for all Q | Extreme save |
| **GQA** | KV groups shared by Q | **Llama 3.2 1B: 32Q / 8KV** |
| MLA | Compress KV to a latent | DeepSeek-style (later chapter) |

### 4. What labs ship

Llama 3.2 1B uses GQA. Many open 7B–70B models use GQA for decode memory.

### 5. What “good” means

- Correct `n_rep = n_q / n_kv`  
- Causal mask / `is_causal` right for prefill vs decode  
- Cache smaller than full MHA without collapsing quality entirely  

### 6. What we do

`CausalSelfAttention` in `arche/attention.py`. Lab prints head counts and shapes.

## Run

```bash
python chapters/01-inference/ep06_gqa.py
```

## Next

**SwiGLU** (`?c=swiglu`): the feed-forward half.
