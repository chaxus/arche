## Idea

This course trains **from zero toward frontier-ready**. Reading order: build the dense trunk → **train weights from zero** → serve faster → frontier structures (MoE/MLA). Wrong specimen ⇒ later “how industry does it” won’t match.

### 1. What it is for

Specimen: **Llama 3.2-style** next-token stack—tokenize → embed → blocks×N → LM head (**dense decoder**). Later chapters hang here.

### 2. From zero: how you “generate” this path

| Step | What you do |
| --- | --- |
| **1. Pick open, downloadable, small, production-like** | Here: Llama 3.2 ~1B |
| **2. Freeze tokenizer; record V, D, depth, heads** | Config is the contract |
| **3. Hand-write the same bricks** | RMSNorm, RoPE, GQA, SwiGLU… |
| **4. Random shapes first, then load the specimen or Part 2 self-train** | Graph → numbers |
| **5. Generate + HF check → train → serve → frontier** | See the four sidebar parts |

### 3. What “good choice” means

| Kind | Check |
| --- | --- |
| **Hard** | Config consistent; bricks match the claimed recipe |
| **Hard** | Supports from-zero: tables trainable, net stackable, weights loadable |
| **Soft** | Docs + obtainable weights |
| **Soft** | Room for MoE/MLA primers |

### 4. Improve · whole-model impact

Swap specimen (e.g. Qwen) ⇒ retarget tokenizer/embeds/blocks. Wider/deeper ⇒ full retrain. We keep 1B and use primers for larger ideas.

### 5. Industry · open models

Llama / Qwen / DeepSeek ship config + checkpoint + tokenizer; engines like vLLM assume such checkpoints.

### 6. This course

```bash
python chapters/01-inference/ep01_overview.py
```

## Run

```bash
python chapters/01-inference/ep01_overview.py
```

## Next

**Tokenizer** (`/tokenizer/`): generate the tokenizer table from zero.
