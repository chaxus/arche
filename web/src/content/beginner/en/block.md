## Idea

```
x → RMSNorm → Attn → +x → RMSNorm → SwiGLU → +x
```

Same shape in/out → stack by looping (1B: 16×).

## Run

```bash
python chapters/01-inference/ep08_block.py
```

## Next

**Stack + LM head** (`?c=stack`).
