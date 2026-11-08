## In one sentence
Embed → N blocks → final RMSNorm → linear to vocabulary scores (often weight-tied).

## Before you start
- Block done. Time: ~12 minutes.

## The idea (plain language)
Glue the pieces:
`ids → Embedding → [Block × N] → RMSNorm → Linear(vocab)`.
The last linear’s scores are logits for the next token. Llama 3.2 ties `lm_head` to embeddings.

## In the code
- `arche/model.py`
- `chapters/01-inference/ep09_stack.py`

## Do this now
```bash
python chapters/01-inference/ep09_stack.py
```

## If something breaks
- Vocab size mismatches between embedding and head → use the provided config helpers.

## Don't worry about yet
Loading real Meta weights (next chapter) or sampling tricks.

## Next
**Load weights** (`?c=weights`).
