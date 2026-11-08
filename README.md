# arche

**ἀρχή** — origin, first principle.

Hand-write a **modern dense** decoder (Llama 3.2 lineage) in PyTorch: RMSNorm, RoPE, SwiGLU, GQA. Load open weights. No GPT-2 archaeology.

- Architecture target: **Llama 3.2 1B** (same family as 3B)
- Pedagogy: one runnable chapter per module
- Learning site: **[arche.chaxus.com](https://arche.chaxus.com)** (ranui)
- License: [Apache-2.0](LICENSE)

## Why not GPT-2

GPT-2 is a fine teaching fossil. This repo starts on the **current trunk**:

| Prefer (here) | Avoid as the main path |
| --- | --- |
| RMSNorm | LayerNorm-as-default story |
| RoPE (+ long-context scaling later) | Absolute learned position embeds |
| SwiGLU | GELU FFN as the “only” FFN |
| GQA | Multi-head without KV grouping |

Frontier MoE / MLA chapters are reserved for later — same repo, not a rewrite.

## Quick start

```bash
git clone https://github.com/chaxus/arche.git
cd arche
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a chapter (each script is standalone):

```bash
python chapters/01-inference/ep01_overview.py
python chapters/01-inference/ep03_embedding.py
python chapters/01-inference/ep04_rmsnorm.py
python chapters/01-inference/ep05_rope.py
python chapters/01-inference/ep06_gqa.py
python chapters/01-inference/ep07_swiglu.py
python chapters/01-inference/ep08_block.py
```

Tokenizer chapter needs the official Llama 3.2 tokenizer (Hugging Face gated model). After you accept the license on the model page:

```bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
```

Mainland network tip:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Learning site

Browse chapters with ranui (`r-theme-switch`, `r-card`, `r-markdown`, `r-math`):

- Production: https://arche.chaxus.com
- Local:

```bash
cd web
npm install
npm run dev
```

Deploy (Cloudflare Pages project `arche`):

```bash
cd web && npm run build
npx wrangler pages deploy dist --project-name=arche --branch=main
```

## Roadmap

**Inference (modern dense)**

| # | Topic | Script |
| --- | --- | --- |
| 01 | Overview & tensor conventions | [ep01](chapters/01-inference/ep01_overview.py) |
| 02 | Tokenizer | [ep02](chapters/01-inference/ep02_tokenizer.py) |
| 03 | Embedding | [ep03](chapters/01-inference/ep03_embedding.py) |
| 04 | RMSNorm | [ep04](chapters/01-inference/ep04_rmsnorm.py) |
| 05 | RoPE | [ep05](chapters/01-inference/ep05_rope.py) |
| 06 | GQA causal attention | [ep06](chapters/01-inference/ep06_gqa.py) |
| 07 | SwiGLU MLP | [ep07](chapters/01-inference/ep07_swiglu.py) |
| 08 | Transformer block | [ep08](chapters/01-inference/ep08_block.py) |
| 09 | Stack + LM head | _next_ |
| 10 | Load pretrained weights | _next_ |
| 11 | Text generation | _next_ |
| — | MoE / MLA / long-context | outlined later |

Library code that chapters grow into lives under [`arche/`](arche/).

## Tests

```bash
pytest -q
```

## References

- Llama 3.2 model cards on Hugging Face (`meta-llama/Llama-3.2-1B`)
- UI components from [ranui](https://github.com/chaxus/ran)
- Pedagogy inspired by step-by-step “from scratch” tutorials; implementation and prose here are original
