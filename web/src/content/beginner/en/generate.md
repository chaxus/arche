## Idea

From zero: logits aren’t generation until you **loop sample**. Products differ in stopping/sampling, not in having a loop.

### 1. What it is for

Last-position logits → optional temperature/top-k → greedy/sample → append → stop on EOS/length.

### 2. From zero: how you build it

Forward→sample→stop; add KV cache next; chat templates later.

### 3. What “good” means

Stops; past-only; valid ids; sensible sampling; coherence with real weights.

### 4. Improve · whole-model impact

Sampling knobs don’t change weights; stop rules shape UX; constrained decoding for structure; without KV the loop is correct but too slow.

### 5. Industry · open models

Every chat stack loops; vLLM/TGI speed cache/batching. Llama/Qwen APIs expose sampling.

### 6. This course

```bash
python chapters/01-inference/ep11_generate.py
```

## Run

```bash
python chapters/01-inference/ep11_generate.py
```

## Next


**HF align** (`/hf-align/`): verify the handwritten stack against HF.
