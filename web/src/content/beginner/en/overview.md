## Idea

This course is **one path from zero toward frontier-ready**. Reading order: tables → **next-token train** → dense bricks → align → serve → frontier (MoE/MLA). The running example is **Llama 3.2 ~1B**—wrong example recipe ⇒ later “how industry does it” won’t match.

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

### 3. Where DeepSeek fits (same path, later map)

When people say **DeepSeek’s frontier open stack** (V2 / V3 / R1-on-V3), they usually mean **not** “another Llama dense.” Headline:

| | **Llama 3.2 (this course’s example)** | **DeepSeek-V2 / V3 line** |
| --- | --- | --- |
| Capacity | **Dense** FFN every layer | **MoE** — many experts, top-k active |
| Attention / KV | **GQA** | **MLA** — latent-compressed KV |
| Shared DNA | RMSNorm / RoPE / modern decoder shape | Same Transformer family, different upgrades |
| In this course | Parts 1–3 hand-write and run | Part 4 primers: MoE → MLA → checkpoint map |

**Read it as:** Llama teaches the dense trunk; DeepSeek is a major open **MoE + MLA** branch on top of that family. We do not start from DeepSeek day one—you need the dense stack first.

### 4. From zero: how you “generate” this path

| Step | What you do |
| --- | --- |
| **1. Fix the example** | Llama 3.2 ~1B |
| **2. Tokenizer + embedding** | Teach-BPE + lookup table; record V, D, depth, heads |
| **3. Next-token train** | Shift + CE + backprop—**writes** embeds and weights |
| **4. Hand-write the bricks** | RMSNorm, RoPE, GQA, SwiGLU… then stack |
| **5. Load Llama / generate / HF → align → serve → frontier** | One sidebar path |

### 5. What “good example” means

| Kind | Check |
| --- | --- |
| **Hard** | Config consistent; bricks match the claimed recipe |
| **Hard** | Supports from-zero: tables trainable, net stackable, weights loadable |
| **Soft** | Docs + obtainable weights |
| **Soft** | Room for MoE/MLA primers (DeepSeek-shaped map) |

### 6. Improve · whole-model impact

Swap the example (e.g. Qwen) ⇒ retarget tokenizer/embeds/blocks. Wider/deeper ⇒ full retrain. We keep 1B and use primers for larger ideas.

### 7. Industry · open models

Llama / Qwen ship dense (and some MoE variants); **DeepSeek-V3** is the open flagship of **large MoE + MLA**. All ship config + checkpoint + tokenizer; engines like vLLM assume such checkpoints.

### 8. This course

```bash
python chapters/01-inference/ep01_overview.py
```

## Run

```bash
python chapters/01-inference/ep01_overview.py
```

## Next

**Tokenizer** (`/tokenizer/`): generate the tokenizer table from zero.
