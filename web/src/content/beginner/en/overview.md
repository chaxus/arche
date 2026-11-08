## Idea

This course is **one path from zero toward frontier-ready**. Reading order: build the dense trunk → **train weights from zero** → serve faster → frontier structures (MoE/MLA). The running example is **Llama 3.2 ~1B**—wrong example recipe ⇒ later “how industry does it” won’t match.

### 1. What it is for

You hand-write a **Llama 3.2-style** next-token stack—tokenize → embed → blocks×N → LM head (**dense decoder**). Later chapters hang on that same spine. The official Llama files are the example’s finished pair, not a second curriculum.

### 2. Why Llama 3.2 ~1B as the example

We needed **one** concrete model to pin config numbers (V, D, depth, heads) and brick names. Criteria → pick:

| Need | Why Llama 3.2 ~1B fits |
| --- | --- |
| **Open + downloadable** | HF checkpoint + official tokenizer; you can load and HF-align |
| **Small enough to hold in your head** | ~1B: depth/heads/`D=2048` stay countable—not a 70B wall of configs |
| **Production-like dense** | RMSNorm, RoPE, GQA, SwiGLU, byte BPE—still what 2024+ dense stacks use (not GPT-2 archaeology) |
| **Finished pair available** | Tokenizer and weights ship together, so “handwritten vs official” is a fair check |
| **Dense first** | MoE/MLA stay later primers; the example itself stays a plain dense decoder |

Not chosen for “best chatbot” or “train a frontier model on a laptop.” Chosen so **from-zero writing** and **industry-shaped reading** share one recipe.

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
