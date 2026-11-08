## Idea

### 1. What it is for

**Frontier checkpoint primer**: save/load MoE/MLA-style modules—round-trip a teaching checkpoint and map Mixtral-like `w1/w2/w3` names into our modules so shapes don’t silently break.

### 2. Why the LLM pipeline needs it

Running MoE/MLA forward isn’t enough; open weights arrive as foreign key names and shards. Can’t load ⇒ frontier structure stays on paper.

### 3. Where it comes from · how you build it

Mapping + round-trip for open MoE-style checkpoints—not a full trainer.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Round-trip `allclose` / shapes | No silent mis-load |
| **Hard** | Name map covers experts | Missed map = random expert |
| **Soft** | Hooks real open weights | Product engineering; shapes here |

### 5. What we do

This episode’s primer script.

## Run

```bash
python chapters/02-training/ep20_frontier_ckpt.py
```

## Next

**HF align** (`?c=hf-align`).
