## Idea

Naive generate re-runs attention over **all** past tokens every step. That repeats the same K/V work for the prompt.

**KV cache:** on the first pass (**prefill**), store every layer’s keys and values. Each new token (**decode**) only computes K/V for itself and **appends** to the cache, then attends over the full cached sequence.

Same answers (in exact arithmetic), far less work as the reply grows. See the interactive strip on this page.

## Run

```bash
python chapters/01-inference/ep12_kv_cache.py
```

Expect a comparison or demo showing cached decode stepping.

## Next

**RoPE scaling** (`?c=rope-scaling`): keep RoPE sane for long prompts.
