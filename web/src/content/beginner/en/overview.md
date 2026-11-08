## Idea

This course is **one path from zero toward frontier-ready**. Reading order: build the dense trunk → **train weights from zero** → serve faster → frontier structures (MoE/MLA). The running example is **Llama 3.2 ~1B**—wrong example recipe ⇒ later “how industry does it” won’t match.

### 1. What it is for

You hand-write a **Llama 3.2-style** next-token stack—tokenize → embed → blocks×N → LM head (**dense decoder**). Later chapters hang on that same spine. The official Llama files are the example’s finished pair, not a second curriculum.

### 2. Why Llama 3.2 ~1B as the example

**Headline:** learn on a **frontier dense architecture**, at a size you can actually finish.

Llama 3.2 is the small open release of Meta’s **Llama 3** line—the same recipe family used in larger Llama (and echoed across Qwen-class dense stacks): **RMSNorm + RoPE + GQA + SwiGLU + byte-level BPE**. That is today’s mainstream open dense trunk, not a simplified “course-only” net. MoE / MLA and bigger engines come later as primers on top of this trunk.

| Why this pick | Plain meaning |
| --- | --- |
| **Frontier-shaped bricks** | Same module names and contracts you meet in current open dense models and serving stacks |
| **One family, teachable scale** | ~1B keeps V / D / depth / heads countable and HF-downloadable—architecture stays Llama 3, size is for learning |
| **Open finished pair** | Official tokenizer + checkpoint so handwritten vs industry is a real check |
| **Dense first** | Master the dense decoder before MoE/MLA maps; don’t start on a museum GPT-2 |

So: **architecture = current frontier dense; scale = ~1B so from-zero is finishable.** Not “the smartest chatbot,” and not “train 70B at home.”

### 3. From zero: how you “generate” this path

| Step | What you do |
| --- | --- |
| **1. Fix the example** | Llama 3.2 ~1B |
| **2. Learn how its kind of tokenizer/embeds are made** | Teach-BPE + toy embed; record V, D, depth, heads |
| **3. Hand-write the same bricks** | RMSNorm, RoPE, GQA, SwiGLU… |
| **4. Random shapes first, then train (Part 2) and/or load the Llama example to check** | Graph → numbers |
| **5. Generate + HF check → serve → frontier** | One sidebar path |

### 4. What “good example” means

| Kind | Check |
| --- | --- |
| **Hard** | Config consistent; bricks match the claimed recipe |
| **Hard** | Supports from-zero: tables trainable, net stackable, weights loadable |
| **Soft** | Docs + obtainable weights |
| **Soft** | Room for MoE/MLA primers |

### 5. Improve · whole-model impact

Swap the example (e.g. Qwen) ⇒ retarget tokenizer/embeds/blocks. Wider/deeper ⇒ full retrain. We keep 1B and use primers for larger ideas.

### 6. Industry · open models

Llama / Qwen / DeepSeek ship config + checkpoint + tokenizer; engines like vLLM assume such checkpoints.

### 7. This course

```bash
python chapters/01-inference/ep01_overview.py
```

## Run

```bash
python chapters/01-inference/ep01_overview.py
```

## Next

**Tokenizer** (`/tokenizer/`): generate the tokenizer table from zero.
