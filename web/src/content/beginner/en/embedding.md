## In one sentence
Each token id becomes a learned list of numbers—a row in a big lookup table.

## Before you start
- Tokenizer concept clear (ids exist).
- Time: ~12 minutes.

## The idea (plain language)
Think of a dictionary: token id → vector of length `embed_dim`. That table is **TokenEmbedding**.

Modern dense models usually **do not** add a separate absolute position table here. Position arrives later via **RoPE** on attention queries/keys.

Llama 3.2 also **ties** embedding weights with the final LM head (same numbers reused).

## Picture / formula
$$X = \mathrm{Embed}(t_{1:L}) \in \mathbb{R}^{L \times d}$$

## In the code
- `arche/embedding.py`
- Companion: `chapters/01-inference/ep03_embedding.py`

## Do this now
```bash
python chapters/01-inference/ep03_embedding.py
```
Look for output shapes like `(batch, seq, embed_dim)`.

## If something breaks
- Shape errors usually mean you edited the demo sizes—re-copy the script from git.

## Don't worry about yet
How embeddings are trained, or fancy token types (segment ids).

## Next
**RMSNorm** (`?c=rmsnorm`).
