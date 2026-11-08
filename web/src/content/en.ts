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
    markdown: `## In one sentence
You will see the big picture of a modern dense language model—and why this course does not start with GPT-2.

## Before you start
- Finish Part 0 (especially the journey map).
- Time: about 12 minutes.
- Optional: run the companion script after reading.

## The idea (plain language)
A **dense decoder** reads tokens from left to right and predicts the next one. Inside, the same kind of block repeats many times. Modern stacks prefer:

- **RMSNorm** to stabilize numbers
- **RoPE** to encode position by rotating vectors
- **SwiGLU** as the feed-forward recipe
- **GQA** so key/value memory stays smaller while decoding

**Llama 3.2 lineage** means we follow that recipe (1B is the teaching target). Capability later comes from official weights; understanding comes from writing the forward pass.

\`\`\`mermaid
flowchart LR
  tok[Tokenizer] --> emb[Embedding]
  emb --> stack[Block_x_N]
  stack --> head[LM_head]
\`\`\`

## Picture / formula
Keep the diagram above in mind. Formulas appear when a chapter needs them.

## In the code
- Config helpers: \`arche/config.py\` (\`llama32_1b\`)
- Companion: \`chapters/01-inference/ep01_overview.py\` prints sizes (layers, heads, dim)—not chat text.

## Do this now
\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`
You should see numbers for vocab size, layers, and heads.

## If something breaks
- \`No module named arche\` → activate \`.venv\` and run from the repo root.
- Script not found → \`cd\` into the \`arche\` folder first.

## Don't worry about yet
Training giant models, serving millions of users, or every Llama variant.

## Next
**Tokenizer** (\`?c=tokenizer\`).

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
    markdown: `## In one sentence
Text becomes integer **token ids** before the model can think in vectors.

## Before you start
- Overview done.
- Time: ~15 minutes.
- Official Llama 3.2 tokenizer is **gated** on Hugging Face—login only when you run this lab.

## The idea (plain language)
Models do not read letters the way humans do. A **tokenizer** chops text into tokens and maps each to an id in a fixed vocabulary (Llama 3.2: 128256 entries).

We use the **official** tokenizer. This course focuses on the neural trunk, not reinventing compression algorithms (BPE).

## In the code
- \`arche/tokenizer.py\`
- Companion: \`chapters/01-inference/ep02_tokenizer.py\`

## Do this now
\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
\`\`\`
Expect printed token ids for a sample string. If you skip login, continue reading—you can return later.

## If something breaks
- Gated 401/403 → accept the model license on Hugging Face, then login again.
- Slow download → optional \`export HF_ENDPOINT=https://hf-mirror.com\` (mainland networks).

## Don't worry about yet
Training your own tokenizer or Unicode edge cases.

## Next
**Embedding** (\`?c=embedding\`).

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
    markdown: `## In one sentence
Each token id becomes a learned list of numbers—a row in a big lookup table.

## Before you start
- Tokenizer concept clear (ids exist).
- Time: ~12 minutes.

## The idea (plain language)
Think of a dictionary: token id → vector of length \`embed_dim\`. That table is **TokenEmbedding**.

Modern dense models usually **do not** add a separate absolute position table here. Position arrives later via **RoPE** on attention queries/keys.

Llama 3.2 also **ties** embedding weights with the final LM head (same numbers reused).

## Picture / formula
$$X = \\mathrm{Embed}(t_{1:L}) \\in \\mathbb{R}^{L \\times d}$$

## In the code
- \`arche/embedding.py\`
- Companion: \`chapters/01-inference/ep03_embedding.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`
Look for output shapes like \`(batch, seq, embed_dim)\`.

## If something breaks
- Shape errors usually mean you edited the demo sizes—re-copy the script from git.

## Don't worry about yet
How embeddings are trained, or fancy token types (segment ids).

## Next
**RMSNorm** (\`?c=rmsnorm\`).

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
    markdown: `## In one sentence
RMSNorm keeps layer inputs well-scaled without the extra mean-centering of LayerNorm.

## Before you start
- Embedding done. Time: ~12 minutes.

## The idea (plain language)
Deep stacks multiply numbers again and again. Without normalization, values can explode or vanish.
**RMSNorm** divides by a root-mean-square size, then scales with a learned gain. It skips subtracting the mean—cheaper, still stable enough for modern LLMs.
Inside each block it runs **before** attention and **before** the MLP (pre-norm).

## Picture / formula
$$\\mathrm{RMSNorm}(x) = \\frac{x}{\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}} \\odot \\gamma$$

## In the code
- \`arche/norms.py\`
- \`chapters/01-inference/ep04_rmsnorm.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## If something breaks
- NaNs in experiments usually mean epsilon or inputs were edited—restore the script.

## Don't worry about yet
Every LayerNorm variant in the literature.

## Next
**RoPE** (\`?c=rope\`).

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
    markdown: `## In one sentence
RoPE rotates query/key vectors so attention “knows” relative positions—no absolute position table required.

## Before you start
- RMSNorm done. Time: ~15 minutes.

## The idea (plain language)
Instead of adding a position vector, **RoPE** rotates pairs of dimensions by an angle that depends on token index. Relative distance falls out of the dot product.
Llama 3.2 also uses a large \`rope_theta\` and later a long-context **scale**—that scale is its own chapter.

## Picture / formula
$$q_m' = R_{\\Theta,m} q_m,\\quad k_n' = R_{\\Theta,n} k_n$$

## In the code
- \`arche/rope.py\`
- \`chapters/01-inference/ep05_rope.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## If something breaks
- Odd head dimensions break pair rotation—use the script’s defaults.

## Don't worry about yet
Every RoPE paper variant; long-context scaling waits for ep13.

## Next
**GQA attention** (\`?c=gqa\`).

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
    markdown: `## In one sentence
Grouped-Query Attention shares key/value heads across several query heads to shrink decode-time memory.

## Before you start
- RoPE idea clear. Time: ~18 minutes.

## The idea (plain language)
Classic multi-head attention gives every query head its own K/V. That is accurate but the **KV cache** gets large while generating.
**GQA** uses fewer KV heads; each KV head is shared by a group of query heads. Llama 3.2 1B: 32 query heads, 8 KV heads.
Flow: project Q/K/V → RoPE(Q,K) → repeat KV → causal attention → output projection.

## In the code
- \`arche/attention.py\`
- \`chapters/01-inference/ep06_gqa.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## If something breaks
- Assert \`num_heads % num_kv_heads == 0\`—groups must divide evenly.

## Don't worry about yet
FlashAttention kernels or MLA (comes later).

## Next
**SwiGLU** (\`?c=swiglu\`).

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
    markdown: `## In one sentence
SwiGLU is the modern feed-forward block: gate × up, then down—three bias-free linears.

## Before you start
- GQA done. Time: ~12 minutes.

## The idea (plain language)
Old tutorials often show Linear → GELU → Linear. Llama-style models prefer:
\`down(silu(gate(x)) * up(x))\`.
The **gate** decides what to pass; **up** proposes values; **down** returns to model width.

## Picture / formula
$$\\mathrm{FFN}(x)=W_d\\big(\\mathrm{SiLU}(W_g x)\\odot W_u x\\big)$$

## In the code
- \`arche/mlp.py\`
- \`chapters/01-inference/ep07_swiglu.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## If something breaks
- Shape mismatch → intermediate_dim must match the three linears in the demo.

## Don't worry about yet
MoE (many experts)—that is Part 2.

## Next
**Transformer block** (\`?c=block\`).

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
    markdown: `## In one sentence
One transformer block is a pre-norm residual sandwich you can stack.

## Before you start
- Attention + SwiGLU done. Time: ~12 minutes.

## The idea (plain language)
\`\`\`
x → RMSNorm → Attn → +x → RMSNorm → SwiGLU → +x
\`\`\`
Same shape in, same shape out—so depth is a loop. Llama 3.2 1B stacks this 16 times (3B: 28).

## In the code
- \`arche/block.py\`
- \`chapters/01-inference/ep08_block.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep08_block.py
\`\`\`

## If something breaks
- Residual add requires matching shapes—don’t change only one branch in experiments.

## Don't worry about yet
Parallel attention/MLP layouts in some giant models.

## Next
**Stack + LM head** (\`?c=stack\`).

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
    markdown: `## In one sentence
Embed → N blocks → final RMSNorm → linear to vocabulary scores (often weight-tied).

## Before you start
- Block done. Time: ~12 minutes.

## The idea (plain language)
Glue the pieces:
\`ids → Embedding → [Block × N] → RMSNorm → Linear(vocab)\`.
The last linear’s scores are logits for the next token. Llama 3.2 ties \`lm_head\` to embeddings.

## In the code
- \`arche/model.py\`
- \`chapters/01-inference/ep09_stack.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

## If something breaks
- Vocab size mismatches between embedding and head → use the provided config helpers.

## Don't worry about yet
Loading real Meta weights (next chapter) or sampling tricks.

## Next
**Load weights** (\`?c=weights\`).

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
    markdown: `## In one sentence
Map official Hugging Face Llama 3.2 tensors into the hand-written \`arche.Llama\` graph.

## Before you start
- Stack chapter done. Time: ~20 minutes.
- Needs Hugging Face login + license acceptance (see Part 0 map).

## The idea (plain language)
Your modules are empty until filled. \`from_pretrained\` builds our graph, downloads HF weights, and copies tensors with a small key map (\`model.layers.*\` → \`layers.*\`).

## In the code
- \`arche/load_weights.py\`
- \`chapters/01-inference/ep10_load_weights.py\`

## Do this now
\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

## If something breaks
- Gated errors → accept license, re-login; try \`HF_ENDPOINT=https://hf-mirror.com\` if needed.
- Out of disk/RAM → close other apps; 1B is the intended size.

## Don't worry about yet
Quantized serving formats or multi-GPU sharding.

## Next
**Text generation** (\`?c=generate\`).

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
    markdown: `## In one sentence
Turn logits into tokens: sample (or greedily pick) the next id, append, repeat until stop.

## Before you start
- Weights chapter understood (even if download deferred). Time: ~15 minutes.

## The idea (plain language)
Each step looks at the last position’s scores. Optional **temperature** sharpens/softens; **top-k** keeps only the best k options. Append the chosen id; stop on EOS or length limit.

## In the code
- \`arche/generate.py\`
- \`chapters/01-inference/ep11_generate.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`
Without weights the tiny demo still shows the loop shape.

## If something breaks
- Empty/gibberish text with random weights is expected—real fluency needs loaded 1B weights.

## Don't worry about yet
KV cache speedups (next part) or chat templates (later).

## Next
Part 2 begins: **KV cache** (\`?c=kv-cache\`).

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
    markdown: `## In one sentence
Prefill the prompt once, then decode new tokens while reusing stored keys and values.

## Before you start
- Generation chapter done. Time: ~15 minutes.

## The idea (plain language)
Without a cache, every new token re-reads the whole prefix in attention—wasteful.
With a **KV cache**: compute K/V for the prompt once, then for each new token only append its K/V and attend against the past.

## In the code
- \`arche/cache.py\`, used by attention + \`generate(..., use_cache=True)\`
- \`chapters/01-inference/ep12_kv_cache.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

## If something breaks
- Wrong \`start_pos\` after edits → restore the chapter script.

## Don't worry about yet
Paged KV (ep23) and production schedulers.

## Next
**RoPE scaling** (\`?c=rope-scaling\`).

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
    markdown: `## In one sentence
Llama 3.2 stretches long wavelengths so context can grow far past the original window.

## Before you start
- RoPE + KV cache ideas clear. Time: ~15 minutes.

## The idea (plain language)
A plain RoPE trained for shorter context misbehaves if you suddenly ask for 100k tokens.
Llama 3.2 keeps a large \`rope_theta\` and applies a **factor / low / high** schedule: short wavelengths stay, long ones stretch, middle band smooths.

## In the code
- Scaling lives in \`arche/rope.py\` / attention config
- \`chapters/01-inference/ep13_rope_scaling.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

## If something breaks
- Tiny demos may not show 128k—look at printed frequency changes instead.

## Don't worry about yet
YaRN vs NTK debates; we teach the Llama 3.2 schedule used here.

## Next
**MoE** (\`?c=moe\`).

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
    markdown: `## In one sentence
Mixture-of-Experts runs only the top-k expert MLPs per token—sparse compute, dense parameter count.

## Before you start
- SwiGLU understood. Time: ~15 minutes.

## The idea (plain language)
A **router** scores experts; each token picks \`top_k\` and blends their outputs.
You can store many experts while each token only pays for a few—ideas behind Mixtral-style stacks.

## In the code
- \`arche/moe.py\`
- \`chapters/01-inference/ep14_moe.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

## If something breaks
- \`top_k > num_experts\` asserts—keep demo defaults.

## Don't worry about yet
Load-balancing losses and production expert parallelism.

## Next
**MLA** (\`?c=mla\`).

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
    markdown: `## In one sentence
Multi-head Latent Attention compresses KV into a small latent cache (DeepSeek-style).

## Before you start
- KV cache intuition. Time: ~18 minutes.

## The idea (plain language)
Instead of caching full per-head K/V, **MLA** stores a low-rank latent (+ a small RoPE stream) and expands when attending.
Same attention idea, cheaper memory at decode.

## In the code
- \`arche/mla.py\`
- \`chapters/01-inference/ep15_mla.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

## If something breaks
- Rank/head_dim mismatches → use the chapter’s constructor args.

## Don't worry about yet
Exact DeepSeek production kernels—key maps come in Part 4.

## Next
**Device · INT8 · chat** (\`?c=device-quant-chat\`).

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
    markdown: `## In one sentence
Pick a device, optionally store weights as INT8, and render chat turns with a template.

## Before you start
- Comfortable running scripts. Time: ~15 minutes.

## The idea (plain language)
\`pick_device\` prefers CUDA → MPS → CPU.
**INT8 weight-only** keeps weights small and dequantizes for compute (teaching grade, not bitsandbytes).
**Chat templates** format role/content; HF template if present, else a plain fallback.

## In the code
- \`arche/device.py\`, \`quantize.py\`, \`chat.py\`
- \`chapters/01-inference/ep16_device_quant_chat.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`
Try the on-page demo too.

## If something breaks
- MPS/CUDA errors → force CPU: the helper accepts \`"cpu"\`.

## Don't worry about yet
GPTQ/AWQ production quant stacks.

## Next
**Continuous batching** (\`?c=continuous-batching\`).

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
    markdown: `## In one sentence
A teaching scheduler steps many short decode jobs together until each finishes.

## Before you start
- Generation + KV ideas. Time: ~15 minutes.

## The idea (plain language)
Production servers do not run one request after another only. They keep a pool of active sequences, step them together, and retire finished ones.
Our version pads active rows each step—**not** paged attention, but the control-loop shape.

## In the code
- \`arche/batching.py\`
- \`chapters/01-inference/ep17_continuous_batching.py\`

## Do this now
\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`
Use the site demo to watch fake jobs share steps.

## If something breaks
- Empty outputs → ensure \`max_new_tokens >= 1\`.

## Don't worry about yet
CUDA graphs + paged fusion (ep23 sketches the pieces).

## Next
Part 3: **Pretrain** (\`?c=pretrain\`).

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
