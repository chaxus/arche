## Idea

Full model path:

`token ids → Embedding → Block×N → final RMSNorm → Linear(vocab)`

The last linear is the **LM head**: each position gets a score per vocab entry (**logits**). Highest (or sampled) id = next token.

Llama 3.2 often **ties** LM head weights with the embedding table—one matrix, two roles.

## Run

```bash
python chapters/01-inference/ep09_stack.py
```

Expect a tiny random model forward and logits shape `(batch, seq, vocab)`.

## Next

**Load weights** (`?c=weights`): swap random init for real Llama tensors.
