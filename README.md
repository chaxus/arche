# arche

**ἀρχή** — origin, first principle.

Hand-write a **modern dense** decoder (Llama 3.2 lineage) in PyTorch: RMSNorm, RoPE, SwiGLU, GQA. Load open weights. No GPT-2 archaeology.

- Architecture target: **Llama 3.2 1B** (same family as 3B)
- Pedagogy: one runnable chapter per module
- Learning site: **[arche.chaxus.com](https://arche.chaxus.com)** (ranui, EN / 中文)
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
python chapters/01-inference/ep09_stack.py
python chapters/01-inference/ep12_kv_cache.py
python chapters/01-inference/ep13_rope_scaling.py
```

Tokenizer / weights / generation need the official Llama 3.2 artifacts (Hugging Face gated). After you accept the license:

```bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
python chapters/01-inference/ep10_load_weights.py
python chapters/01-inference/ep11_generate.py
```


Mainland network tip:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Library entry points:

```python
from arche import from_pretrained, generate
from arche.tokenizer import load_tokenizer
import torch

tok = load_tokenizer()
model = from_pretrained("meta-llama/Llama-3.2-1B")
ids = torch.tensor([tok.encode("Hello")])
print(tok.decode(generate(model, ids, max_new_tokens=20)[0].tolist()))
```

## Learning site

Browse chapters with ranui (`r-select`, `r-theme-switch`, `r-card`, `r-markdown`, `r-math`).
Locales follow the same URL shape as [ran.chaxus.com](https://ran.chaxus.com): English at `/`, Chinese under `/cn/`.

- English: https://arche.chaxus.com/
- 简体中文：https://arche.chaxus.com/cn/
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
| 09 | Stack + LM head | [ep09](chapters/01-inference/ep09_stack.py) |
| 10 | Load pretrained weights | [ep10](chapters/01-inference/ep10_load_weights.py) |
| 11 | Text generation | [ep11](chapters/01-inference/ep11_generate.py) |
| 12 | KV cache | [ep12](chapters/01-inference/ep12_kv_cache.py) |
| 13 | RoPE long-context scaling | [ep13](chapters/01-inference/ep13_rope_scaling.py) |
| — | MoE / MLA | outlined later |

Library code that chapters grow into lives under [`arche/`](arche/).

## Tests

```bash
pytest -q
```

## References

- Llama 3.2 model cards on Hugging Face (`meta-llama/Llama-3.2-1B`)
- UI components from [ranui](https://github.com/chaxus/ran)
- Pedagogy inspired by step-by-step “from scratch” tutorials; implementation and prose here are original
