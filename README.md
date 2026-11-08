# arche

**ἀρχή** — origin, first principle.

**From zero toward frontier-ready:** see what a chatbot computes—by writing one. Build the dense trunk, **train (or load) weights**, serve faster, then map MoE/MLA-style tips. Specimen: small **Llama 3.2** (~1B). Site: **[arche.chaxus.com](https://arche.chaxus.com)** (EN / 中文).

从零对接前沿：亲手写出主干 → 训练（或加载）权重 → 服务加速 → 前沿结构地图。学习站：[arche.chaxus.com/cn/](https://arche.chaxus.com/cn/)。

| Plain path | What’s inside |
| --- | --- |
| **1 Build the dense trunk** | Tokenizer/embedding from scratch (or official pair) → RMSNorm, RoPE, GQA, SwiGLU, stack → load/generate → HF align |
| **2 Train from zero** | Toy pretrain, SFT, DPO — how weights get written |
| **3 Generate & serve faster** | KV cache, RoPE scaling, device/quant/chat, continuous batching, paged KV / CUDA Graph |
| **4 Frontier structures** | MoE, MLA, frontier checkpoints, sharded loads |

**Two paths:** (A) official Llama tokenizer + weights to verify the stack; (B) teach-BPE + toy embed/pretrain to feel from-zero generation. Interleave as you like.

In-course = runnable principles + a map into tip papers and engines. Full production stacks (vLLM-class, huge trainers, multimodal) are the **next** climb—with that map, not from zero.

**New here?** Open the site → **Continue** / Part 0, then `python chapters/00-start/ep00_check_setup.py`.

- License: [Apache-2.0](LICENSE)

## Why Llama 3.2

Small enough to hold in your head (**~1B**), downloadable HF weights, and the same parts production dense models still use—then newer tricks (MoE / MLA / …) once the base path is yours.

## Quick start

```bash
git clone https://github.com/chaxus/arche.git
cd arche
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run chapters (each script is standalone). Order below matches the **from-zero site path** (train before heavy serving):

```bash
python chapters/00-start/ep00_check_setup.py
python chapters/01-inference/ep01_overview.py
python chapters/01-inference/ep02_teach_bpe.py          # train a tiny tokenizer table (no HF)
python chapters/01-inference/ep03_embedding.py          # gather + one toy train step
python chapters/01-inference/ep04_rmsnorm.py
python chapters/01-inference/ep05_rope.py
python chapters/01-inference/ep06_gqa.py
python chapters/01-inference/ep07_swiglu.py
python chapters/01-inference/ep08_block.py
python chapters/01-inference/ep09_stack.py
python chapters/02-training/ep18_pretrain.py            # Part 2 — write weights (toy)
python chapters/02-training/ep19_sft_dpo.py
python chapters/01-inference/ep12_kv_cache.py           # Part 3 — serve
python chapters/01-inference/ep13_rope_scaling.py
python chapters/01-inference/ep16_device_quant_chat.py
python chapters/01-inference/ep17_continuous_batching.py
python chapters/01-inference/ep23_paged_cuda_graph.py
python chapters/01-inference/ep14_moe.py                # Part 4 — frontier
python chapters/01-inference/ep15_mla.py
python chapters/02-training/ep20_frontier_ckpt.py
python chapters/02-training/ep22_sharded_frontier.py
```

Tokenizer / weights / generation / HF align need the official Llama 3.2 artifacts (Hugging Face gated). After you accept the license:

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

## Site

```bash
cd web && npm install && npm run dev
```

Open the printed local URL (CN: `/cn/`). Content lives in `web/src/content/beginner/{en,cn}/`; `python tools/apply_beginner_md.py` injects into `en.ts` / `cn.ts`.
