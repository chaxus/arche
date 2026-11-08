## Idea

### 1. Why stack

One block isn’t a language model—many blocks + a vocab head are.

### 2. Full path

`ids → Embed → Block×N → final RMSNorm → Linear(vocab)`

Last scores = **logits** for the next token. Often **tie** LM head ↔ embedding.

### 3. Pieces

| Piece | Role |
| --- | --- |
| Embedding | ids → vectors |
| Blocks | contextualize |
| Final norm | stabilize before head |
| LM head | vectors → vocab scores |

### 4. What labs ship

Llama-style dense decoder end-to-end (still random weights until the load chapter).

### 5. What “good” means

- Logits shape `(B, T, vocab)`  
- Tie setting matches the config you claim  

### 6. What we do

`Llama` in `arche/model.py`.

## Run

```bash
python chapters/01-inference/ep09_stack.py
```

## Next

**Load weights** (`?c=weights`).
