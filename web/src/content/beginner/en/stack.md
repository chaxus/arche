## Idea

### 1. What it is for

The **stack** wires embed → `N` blocks → final RMSNorm → LM head into a full next-token model:

`ids → embed → Block×N → final RMSNorm → Linear(vocab) → logits`

Last-dim scores are next-token **logits**. Small Llamas often **tie** LM head ↔ embedding.

### 2. Why the LLM pipeline needs it

One block isn’t a LM. Depth mixes context; the LM head maps hidden states back to **the same vocab** as the tokenizer so you can sample the next id.

### 3. Where it comes from · how you build it

| Part | Role |
| --- | --- |
| Embed | id → vector |
| Blocks × N | mix + transform |
| Final norm | steady before the head |
| LM head | vector → vocab scores (maybe tied) |

Config sets depth/dim/heads; random weights mean a correct graph but nonsense outputs (next chapter loads).

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | logits `(B, T, vocab)` | Align with vocab |
| **Hard** | Tying matches claim | Shared params if tied |
| **Hard** | Depth/dim match config | Else official weights won’t fit |
| **Soft** | Coherent text after load | Full-model quality |

### 5. What labs ship · what we do

Llama-style dense end-to-end.  
`Llama` in `arche/model.py`.

## Run

```bash
python chapters/01-inference/ep09_stack.py
```

## Next

**Load weights** (`?c=weights`).
