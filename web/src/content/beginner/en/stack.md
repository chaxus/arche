## Idea

`ids → Embed → [Block×N] → RMSNorm → Linear(vocab)`. Last scores = next-token logits. LM head tied to embeddings.

## Run

```bash
python chapters/01-inference/ep09_stack.py
```

## Next

**Load weights** (`?c=weights`).
