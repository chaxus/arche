## Idea

From zero: one block isn’t an LM—you need **embed + N blocks + final norm + LM head**.

### 1. What it is for

`ids → embed → Block×N → final RMSNorm → Linear(V) → logits`. Small Llamas often **tie** head↔embed.

### 2. From zero: how you build it

Build shell from V/D/N/heads; wire stack; choose tying; init then load or pretrain.

### 3. What “good” means

Logits `(B,T,V)`; config matches weights; tying consistent; quality after load/train.

### 4. Improve · whole-model impact

Depth/width/tie/V are architecture-level—retrain or careful migration. Optimize modules + training, not “stack alone.”

### 5. Industry · open models

Llama/Qwen dense share this skeleton; MoE swaps FFN for experts inside blocks.

### 6. This course

```bash
python chapters/01-inference/ep09_stack.py
```

## Run

```bash
python chapters/01-inference/ep09_stack.py
```

## Next

**Load weights** (`?c=weights`).
