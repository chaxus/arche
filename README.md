# arche

**ἀρχή** — origin, first principle.

Hand-write a **modern dense** decoder (Llama 3.2 lineage) in PyTorch, then the frontier add-ons that matter next: **KV cache**, **RoPE scaling**, **MoE**, **MLA**, plus inference engineering and tiny **pretrain / SFT / DPO** loops. Load open weights. No GPT-2 archaeology.

- Architecture target: **Llama 3.2 1B** (same family as 3B), plus MoE / MLA primers
- Pedagogy: one runnable chapter per module
- Learning site: **[arche.chaxus.com](https://arche.chaxus.com)** (ranui, EN / 中文，chapter progress + demos)
- License: [Apache-2.0](LICENSE)

## Why not GPT-2

GPT-2 is a fine teaching fossil. This repo starts on the **current trunk**:

| Prefer (here) | Avoid as the main path |
| --- | --- |
| RMSNorm | LayerNorm-as-default story |
| RoPE (+ long-context scaling) | Absolute learned position embeds |
| SwiGLU / MoE | GELU FFN as the “only” FFN |
| GQA / MLA | Multi-head without KV grouping or compression |

## Quick start

```bash
git clone https://github.com/chaxus/arche.git
cd arche
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run chapters (each script is standalone):

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
python chapters/01-inference/ep14_moe.py
python chapters/01-inference/ep15_mla.py
python chapters/01-inference/ep16_device_quant_chat.py
python chapters/01-inference/ep17_continuous_batching.py
python chapters/02-training/ep18_pretrain.py
python chapters/02-training/ep19_sft_dpo.py
python chapters/02-training/ep20_frontier_ckpt.py
python chapters/01-inference/ep21_hf_align.py
python chapters/02-training/ep22_sharded_frontier.py
python chapters/01-inference/ep23_paged_cuda_graph.py
```

Tokenizer / weights / generation need the official Llama 3.2 artifacts (Hugging Face gated). After you accept the license:

```bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
python chapters/01-inference/ep10_load_weights.py
python chapters/01-inference/ep11_generate.py
python chapters/01-inference/ep21_hf_align.py
```

Optional HF logits alignment (skipped in CI without weights):

```bash
ARCH_HF_ALIGN=1 pytest tests/test_align_hf.py -q
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

Automatic deploy on `main` (workflow `.github/workflows/deploy-pages.yml`) needs repo secrets:

- `CLOUDFLARE_API_TOKEN` — Pages edit permission
- `CLOUDFLARE_ACCOUNT_ID` — from `wrangler whoami`

Without those secrets the deploy job is skipped; CI tests still run.

## Curriculum

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
| 14 | MoE (top-k experts) | [ep14](chapters/01-inference/ep14_moe.py) |
| 15 | MLA (latent KV) | [ep15](chapters/01-inference/ep15_mla.py) |
| 16 | Device · INT8 · chat template | [ep16](chapters/01-inference/ep16_device_quant_chat.py) |
| 17 | Continuous batching (teaching) | [ep17](chapters/01-inference/ep17_continuous_batching.py) |
| 18 | Pretrain toy loop | [ep18](chapters/02-training/ep18_pretrain.py) |
| 19 | SFT · DPO | [ep19](chapters/02-training/ep19_sft_dpo.py) |
| 20 | Frontier MoE/MLA checkpoints | [ep20](chapters/02-training/ep20_frontier_ckpt.py) |
| 21 | HF logits alignment | [ep21](chapters/01-inference/ep21_hf_align.py) |
| 22 | Sharded Mixtral / DeepSeek MLA | [ep22](chapters/02-training/ep22_sharded_frontier.py) |
| 23 | Paged KV · CUDA graphs | [ep23](chapters/01-inference/ep23_paged_cuda_graph.py) |

Library code lives under [`arche/`](arche/).

## Tests

```bash
pytest -q -m "not integration"
```

CI runs the same gate on every push / PR (Python + learning-site build).

## References

- Llama 3.2 model cards on Hugging Face (`meta-llama/Llama-3.2-1B`)
- Mixtral / DeepSeek papers for MoE and MLA intuition
- UI components from [ranui](https://github.com/chaxus/ran)
- Pedagogy inspired by step-by-step “from scratch” tutorials; implementation and prose here are original
