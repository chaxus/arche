# arche

**ἀρχή** — origin, first principle.

**From zero: see what a chatbot is computing—by writing one.** Build the model in PyTorch, make it faster, try a little training, then peek at newer tricks (mixture-of-experts, compressed attention, paged memory…). Specimen: small **Llama 3.2** (~1B). Site: **[arche.chaxus.com](https://arche.chaxus.com)** (EN / 中文).

从零开始：亲手写出大模型在算什么 → 让它更快 → 练一练 → 看更前沿的招。学习站：[arche.chaxus.com/cn/](https://arche.chaxus.com/cn/)（不要求先懂术语）。

| Plain path | What’s inside |
| --- | --- |
| **1 Build the model** | Write next-token path end to end (RMSNorm, RoPE, attention, SwiGLU, stack, HF weights) |
| **2 Make it faster** | KV cache, long-context RoPE, batching, quant, chat template |
| **3 Train a little** | Toy pretrain, SFT, DPO |
| **4 Newer tricks** | MoE, MLA, sharded checkpoints, paged KV, CUDA Graph |

In-course = runnable principles + a map into tip papers and engines. Full production stacks (vLLM-class, huge trainers, multimodal) are the **next** climb—with that map, not from zero.

**New here?** Open the site → **Continue** / Part 0, then `python chapters/00-start/ep00_check_setup.py`.

- License: [Apache-2.0](LICENSE)

## Why Llama 3.2

Small enough to hold in your head (**~1B**), downloadable HF weights, and the same parts production models still use—then newer tricks (MoE / MLA / …) on top once the base path is yours.

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
python chapters/00-start/ep00_check_setup.py
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

Site Part 0 (start here): welcome → what an LLM does → setup → terminal → reading code → map (`chapters/00-start/ep00_check_setup.py`).

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
