## Idea

Generation is a loop, not one shot:

1. Run the model on the current ids → logits for the **last** position  
2. Optionally soften with **temperature**, keep only **top-k**  
3. **Greedy** (`argmax`) or **sample** from the distribution  
4. Append the new id; stop on EOS or max length  

With a KV cache (next chapter) step 1 becomes cheap after the first prefill. This chapter focuses on the sampling loop itself.

## Run

```bash
python chapters/01-inference/ep11_generate.py
```

Expect a short decoded string (needs weights from the previous chapter for real text; otherwise follow the script’s notes).

## Next

**KV cache** (`?c=kv-cache`): stop recomputing the whole prompt every step.
