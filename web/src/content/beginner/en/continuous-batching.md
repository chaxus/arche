## Idea

### 1. What it is for

**Continuous batching (teaching)**: many short decode jobs share one step loop; finished jobs retire so new ones can enter—less “one loop per user, fast waits for slow.”

Try the page demo. **Not** full paged / vLLM.

### 2. Why the LLM pipeline needs it

Servers serve many users. One loop each → GPU idle when finish times differ. Continuous batching turns “who’s still active this step” into scheduling; real systems also need paged KV and fused kernels.

### 3. Where it comes from · how you build it

| Layer | Role |
| --- | --- |
| **Scheduler** | Who’s active each step (this lesson) |
| **Paging / kernels** | How memory/compute run (later primer) |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Finished seqs retire | Don’t keep stepping them |
| **Hard** | No cross-talk | Batched ≠ mixed semantics |
| **Soft** | Fuller GPU, steadier latency | Serving metrics; shape only here |

### 5. What labs ship · what we do

Teaching schedule shape + this script.

## Run

```bash
python chapters/01-inference/ep17_continuous_batching.py
```

## Next

**Pretrain** (`?c=pretrain`).
