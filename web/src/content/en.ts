import type { Chapter } from './types';
import { PART0_EN } from './part0';

export const CHAPTERS_EN: Chapter[] = [
  ...PART0_EN,
  {
    id: 'overview',
    ep: '01',
    title: 'Overview',
    summary: 'Modern dense trunk — Llama 3.2 lineage, not GPT-2 archaeology.',
    script: 'chapters/01-inference/ep01_overview.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Why this path

**arche** (ἀρχή) means *first principle*. We hand-write a **current** dense decoder:

- **RMSNorm** instead of LayerNorm-as-default
- **RoPE** instead of absolute position tables
- **SwiGLU** instead of a GELU-only FFN story
- **GQA** so decode-time KV cache stays affordable

Target weights: **Llama 3.2 1B** (same family as 3B). Capability comes from open weights; understanding comes from writing the forward pass.

\`\`\`mermaid
flowchart LR
  tok[Tokenizer] --> emb[Embedding]
  emb --> stack[Block_x_N]
  stack --> head[LM_head]
  stack --- rms[RMSNorm]
  stack --- rope[RoPE]
  stack --- gqa[GQA]
  stack --- swiglu[SwiGLU]
\`\`\`
`,
  },
  {
    id: 'tokenizer',
    ep: '02',
    title: 'Tokenizer',
    summary: 'Official Llama 3.2 BPE — text becomes token ids.',
    script: 'chapters/01-inference/ep02_tokenizer.py',
    partId: 'trunk',
    estimatedMinutes: 15,
    markdown: `## Role

The model never sees characters. A tokenizer maps text ↔ integer ids in a fixed vocabulary (\`128256\` for Llama 3.2).

We use the **official** tokenizer (Hugging Face gated model). The learning focus of this repo is the dense trunk, not reinventing BPE.

Run locally:

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
\`\`\`
`,
  },
  {
    id: 'embedding',
    ep: '03',
    title: 'Embedding',
    summary: 'Token ids → dense vectors. No absolute PE table.',
    script: 'chapters/01-inference/ep03_embedding.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Lookup, not position

\`TokenEmbedding\` is a table of shape \`(vocab_size, embed_dim)\`.

Modern dense stacks **do not** add a learned absolute position embedding here. Position enters later through **RoPE** on Q and K.

Llama 3.2 1B/3B also **tie** the embedding weight with the LM head.
`,
    latex: 'X = \\mathrm{Embed}(t_{1:L}) \\in \\mathbb{R}^{L \\times d}',
  },
  {
    id: 'rmsnorm',
    ep: '04',
    title: 'RMSNorm',
    summary: 'Pre-norm without mean centering.',
    script: 'chapters/01-inference/ep04_rmsnorm.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Why RMSNorm

Cheaper than LayerNorm (no mean subtraction), stable enough for modern LLMs, and the default pre-norm in Llama / Qwen dense stacks.

Applied **before** attention and **before** the MLP inside each block.
`,
    latex:
      '\\mathrm{RMSNorm}(x) = \\frac{x}{\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}} \\odot \\gamma',
  },
  {
    id: 'rope',
    ep: '05',
    title: 'RoPE',
    summary: 'Rotate Q/K so attention becomes relative in position.',
    script: 'chapters/01-inference/ep05_rope.py',
    partId: 'trunk',
    estimatedMinutes: 15,
    markdown: `## Rotary positions

RoPE rotates pairs of dimensions in query/key vectors by an angle that depends on token position. Relative offsets fall out of the dot product — no absolute PE table required.

Llama 3.2 also uses a large \`rope_theta\` (\`500000\`) and a long-context **scale factor** (deferred to a later chapter).
`,
    latex:
      'q_m^{\\prime} = R_{\\Theta,m} q_m,\\quad k_n^{\\prime} = R_{\\Theta,n} k_n',
  },
  {
    id: 'gqa',
    ep: '06',
    title: 'GQA attention',
    summary: 'Fewer KV heads than query heads, still causal.',
    script: 'chapters/01-inference/ep06_gqa.py',
    partId: 'trunk',
    estimatedMinutes: 18,
    markdown: `## Grouped-Query Attention

Llama 3.2 1B: **32** query heads, **8** KV heads → each KV head is shared by 4 query heads.

Benefits show up at **decode** time: the KV cache shrinks while quality stays close to full multi-head attention.

Flow inside the module: project Q/K/V → RoPE(Q,K) → repeat KV → causal SDPA → output projection.
`,
  },
  {
    id: 'swiglu',
    ep: '07',
    title: 'SwiGLU',
    summary: 'gate × up, then down — three bias-free linears.',
    script: 'chapters/01-inference/ep07_swiglu.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Modern FFN

\`\`\`
down(silu(gate(x)) * up(x))
\`\`\`

This replaces the older “one Linear → GELU → Linear” story as the default in Llama / Qwen dense models.
`,
    latex: '\\mathrm{FFN}(x)=W_d\\big(\\mathrm{SiLU}(W_g x)\\odot W_u x\\big)',
  },
  {
    id: 'block',
    ep: '08',
    title: 'Transformer block',
    summary: 'Pre-norm residual sandwich you can stack.',
    script: 'chapters/01-inference/ep08_block.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## One block

\`\`\`
x → RMSNorm → Attn → +x → RMSNorm → SwiGLU → +x
\`\`\`

Llama 3.2 1B stacks this **16** times (3B: 28). Same shape in, same shape out — that is why depth is just a loop.
`,
  },
  {
    id: 'stack',
    ep: '09',
    title: 'Stack + LM head',
    summary: 'embed → blocks → final RMSNorm → tied lm_head.',
    script: 'chapters/01-inference/ep09_stack.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Full forward

\`\`\`
ids → Embedding → [Block × N] → RMSNorm → Linear(vocab)
\`\`\`

Llama 3.2 1B/3B **tie** \`lm_head.weight\` with \`embed_tokens.weight\`.

Run:

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`
`,
  },
  {
    id: 'weights',
    ep: '10',
    title: 'Load weights',
    summary: 'Map Hugging Face Llama 3.2 tensors into arche.Llama.',
    script: 'chapters/01-inference/ep10_load_weights.py',
    partId: 'trunk',
    estimatedMinutes: 20,
    markdown: `## Official weights, hand-written graph

\`from_pretrained\` builds our modules, downloads HF weights, and copies tensors by a small key map (\`model.layers.*\` → \`layers.*\`).

Needs a gated HF login:

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`
`,
  },
  {
    id: 'generate',
    ep: '11',
    title: 'Text generation',
    summary: 'Sample the next tokens with temperature and top-k.',
    script: 'chapters/01-inference/ep11_generate.py',
    partId: 'trunk',
    estimatedMinutes: 15,
    markdown: `## Decode loop

Repeatedly take the last-step logits, optionally filter top-k, sample (or greedy), append, stop on EOS.

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

Capability now comes from the loaded 1B weights; the code path is still entirely ours.
`,
  },
  {
    id: 'kv-cache',
    ep: '12',
    title: 'KV cache',
    summary: 'Prefill once, then decode with cached keys and values.',
    script: 'chapters/01-inference/ep12_kv_cache.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## Why cache

Without a cache, every new token re-runs attention over the whole prefix.

With a cache: project only the new token, append its K/V, attend against the stored past.

\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

\`generate(..., use_cache=True)\` is the default.
`,
  },
  {
    id: 'rope-scaling',
    ep: '13',
    title: 'RoPE scaling',
    summary: 'Llama 3.2 wavelength smoothing for long context.',
    script: 'chapters/01-inference/ep13_rope_scaling.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## Beyond the 8k window

Llama 3.2 keeps \`rope_theta=500000\` and applies a **factor / low / high** schedule so long wavelengths stretch toward 128k context.

\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

Short wavelengths stay put; long ones divide by \`scale_factor\`; the middle band is smoothed — same idea as Hugging Face's llama3 RoPE.
`,
  },
  {
    id: 'moe',
    ep: '14',
    title: 'MoE',
    summary: 'Top-k routed SwiGLU experts — sparse capacity.',
    script: 'chapters/01-inference/ep14_moe.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## Mixture of Experts

A linear router scores experts per token; only \`top_k\` run. Parameter count can be huge while per-token FLOPs stay near a dense MLP.

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

This is the sparse FFN idea behind Mixtral and DeepSeek-style frontier stacks.
`,
  },
  {
    id: 'mla',
    ep: '15',
    title: 'MLA',
    summary: 'Compress KV into a latent cache (DeepSeek-style).',
    script: 'chapters/01-inference/ep15_mla.py',
    partId: 'infer',
    estimatedMinutes: 18,
    markdown: `## Multi-head Latent Attention

Instead of caching every head's K/V, MLA stores a low-rank latent (+ a small RoPE stream) and expands on the fly.

\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

That is the decode-memory win DeepSeek popularized — same attention math, cheaper cache.
`,
  },
  {
    id: 'device-quant-chat',
    ep: '16',
    title: 'Device · INT8 · chat',
    summary: 'Pick CUDA/MPS/CPU, weight-only INT8, and chat templates.',
    script: 'chapters/01-inference/ep16_device_quant_chat.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'chat',
    markdown: `## Inference engineering basics

\`pick_device\` prefers CUDA → MPS → CPU. \`quantize_llama_int8\` swaps Linear layers for per-channel int8 + scale (educational, not bitsandbytes).

Chat turns go through \`apply_chat_template\` — HF template when present, a plain role/content fallback otherwise.

\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`

HF logits alignment lives in \`tests/test_align_hf.py\` (skipped without gated weights; force with \`ARCH_HF_ALIGN=1\`).
`,
  },
  {
    id: 'continuous-batching',
    ep: '17',
    title: 'Continuous batching',
    summary: 'Schedule many short decodes in one stepping loop.',
    script: 'chapters/01-inference/ep17_continuous_batching.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'batch',
    markdown: `## Teaching scheduler

Pack active requests each step, sample the next token, retire jobs that hit EOS or \`max_new_tokens\`.

This is **not** paged attention or a production engine — it teaches the control loop shape behind continuous batching.

\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`
`,
  },
  {
    id: 'pretrain',
    ep: '18',
    title: 'Pretrain',
    summary: 'Tiny next-token AdamW loop on toy batches.',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'train',
    estimatedMinutes: 15,
    demo: 'train',
    markdown: `## From forward to learning

Causal LM loss: predict token \`t+1\` from positions \`0..T-1\`. A few AdamW steps on random ids are enough to feel the training API.

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`
`,
  },
  {
    id: 'sft-dpo',
    ep: '19',
    title: 'SFT · DPO',
    summary: 'Mask the prompt for SFT; one preference step for DPO.',
    script: 'chapters/02-training/ep19_sft_dpo.py',
    partId: 'train',
    estimatedMinutes: 18,
    demo: 'train',
    markdown: `## Aligning after pretrain

**SFT** supervises only the response span. **DPO** compares chosen vs rejected log-prob ratios against a frozen reference (Rafailov et al.).

\`\`\`bash
python chapters/02-training/ep19_sft_dpo.py
\`\`\`
`,
  },
  {
    id: 'frontier-ckpt',
    ep: '20',
    title: 'Frontier checkpoints',
    summary: 'MoE/MLA roundtrip + Mixtral-shaped expert key map.',
    script: 'chapters/02-training/ep20_frontier_ckpt.py',
    partId: 'frontier',
    estimatedMinutes: 15,
    markdown: `## Wiring the primers

Save/load MoE and MLA modules with \`save_checkpoint\` / \`load_checkpoint\`. Map Mixtral \`w1/w2/w3\` expert keys into \`MoESwiGLU\`.

Sharded Mixtral / DeepSeek layouts continue in ep22.

\`\`\`bash
python chapters/02-training/ep20_frontier_ckpt.py
\`\`\`
`,
  },
  {
    id: 'hf-align',
    ep: '21',
    title: 'HF logits alignment',
    summary: 'Compare arche vs Hugging Face Llama 3.2 on a short prompt.',
    script: 'chapters/01-inference/ep21_hf_align.py',
    partId: 'frontier',
    estimatedMinutes: 20,
    markdown: `## Prove the hand-written graph

\`compare_logits_to_hf\` loads Meta's gated 1B weights into both stacks and checks argmax + max-abs drift.

Without login / cache the chapter exits cleanly; CI keeps the synthetic bit-exact mapping test.

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
# or: ARCH_HF_ALIGN=1 pytest tests/test_align_hf.py -q
\`\`\`
`,
  },
  {
    id: 'sharded-frontier',
    ep: '22',
    title: 'Sharded MoE / MLA',
    summary: 'index.json + shard files into Mixtral MoE and DeepSeek MLA.',
    script: 'chapters/02-training/ep22_sharded_frontier.py',
    partId: 'frontier',
    estimatedMinutes: 15,
    markdown: `## Same shape as HF indexes

Write synthetic Mixtral / DeepSeek shards, then reload into \`MoESwiGLU\` and \`MultiHeadLatentAttention\` with allclose checks.

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`
`,
  },
  {
    id: 'paged-cuda-graph',
    ep: '23',
    title: 'Paged KV · CUDA graphs',
    summary: 'Block-table KV pool plus fixed-shape CUDA graph capture.',
    script: 'chapters/01-inference/ep23_paged_cuda_graph.py',
    partId: 'frontier',
    estimatedMinutes: 18,
    markdown: `## Toward production decode

**Paged KV** stores tokens in fixed pages and gathers a contiguous cache for SDPA.

**CUDA graphs** capture a fixed-shape module step; Llama decode stays eager until KV writes are in-place (pages), which is what engines graph next.

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`
`,
  },
];
