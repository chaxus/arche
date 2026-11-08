import type { Chapter } from './types';
import { PART0_EN } from './part0';

export const CHAPTERS_EN: Chapter[] = [
  ...PART0_EN,
  {
    id: 'overview',
    ep: '01',
    title: 'Overview',
    summary: 'Why Llama 3.2 as the specimen, and what the full next-token path looks like.',
    script: 'chapters/01-inference/ep01_overview.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Idea

We hand-write a small **Llama 3.2** (~1B parameters): small enough to hold in your head, downloadable weights, same parts used in production. People often call this “layers that all run, then guess the next token” a **dense decoder**—learn the name as you go. Goal: own every block of the forward pass.

\`\`\`mermaid
flowchart LR
  tok[Tokenizer] --> emb[Embedding]
  emb --> stack[Blocks]
  stack --> head[LM head]
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

Prints config sizes (layers, heads, dim)—not chat.

## Next

**Tokenizer** (\`?c=tokenizer\`).

`,
  },
  {
    id: 'tokenizer',
    ep: '02',
    title: 'Tokenizer',
    summary: 'Why tokenize, how labs cut text, how to judge quality, and why the official table.',
    script: 'chapters/01-inference/ep02_tokenizer.py',
    partId: 'trunk',
    estimatedMinutes: 22,
    demo: 'tokenizer',
    markdown: `## Idea

### 1. Why tokenize

Models compute on numbers, not raw strings like “hello world”.

- One symbol per whole sentence → vocab explodes; unseen text breaks  
- Pure bytes only → sequences too long (slow, burns context)  

**Tokenization** is the middle ground: stable, learnable, length-controlled.  
One line: **same rules encode in and decode out.**

### 2. What a tokenizer does

1. **Split** — text → small chunks (**tokens**)  
2. **Number** — fixed vocab → **token ids** (ints)  

Everything downstream uses ids; generation ends by decoding with the **same** table. Specials (bos/eos/…) are fixed ids too.

\`\`\`mermaid
flowchart LR
  text[Text] --> cut[Tokens]
  cut --> ids[token ids]
  ids --> model[Model]
  model --> ids2[New ids]
  ids2 --> back[Decoded text]
\`\`\`

Panel above: **teaching BPE** (intuition). Lab below: **official Llama 3.2** table.

### 3. Common cuts, and three names

| Cut | Plain idea | Today |
| --- | --- | --- |
| Word / whitespace | Spaces; CJK needs help | Rare as the main LLM scheme |
| Char / byte | One id per char/byte | Safety floor; pure use is too long |
| **BPE** | Merge frequent adjacent pieces | **Mainstream open decoders** |
| **WordPiece** | Subword, different merge rule | BERT-era |
| **Unigram** (often via **SentencePiece**) | Shrink a large candidate vocab | Still used in some stacks |

**BPE:** start from bytes/chars; merge the most frequent adjacent pair; repeat. Common chunks become their own tokens (\`the\`, \`ing\`).

**WordPiece:** also subword, but merges lean on likelihood, not raw counts. Looks like BPE; **different rule — don’t mix tables.**

**Unigram / SentencePiece:** Unigram = train-by-pruning objective; SentencePiece ≈ **toolkit** (often cuts raw text—handy for CJK). “SentencePiece tokenizer” usually means Unigram (or BPE) trained with that tool.

All three are **subwords**. Product gaps more often come from **data, pretokenizer regex, vocab size**.

### 4. What labs actually ship

| Family | Engine | ~Vocab | Note |
| --- | --- | --- | --- |
| GPT-4 / 3.5 | tiktoken BPE \`cl100k\` | ~100K | Encoding open |
| GPT-4o / o | tiktoken BPE \`o200k\` | ~200K | Larger; often fewer non-English tokens |
| Llama 3 / 3.2 | byte BPE | 128K+ | **This course** |
| DeepSeek-V3 | byte BPE | ~128K | Pretokenizer/data for multilingual compression |
| Kimi K2 | tiktoken-style BPE | ~163K | Han-aware regex |
| Qwen, … | BPE | ~150K | Same road; own merges |

Takeaway: everyone iterates **inside byte-level BPE / tiktoken-family**.

### 5. What “good” means (and what “weights” are)

| Criterion | Plain meaning |
| --- | --- |
| **Compression / fertility** | Tokens for the same text; fewer is tighter. Fertility = tokens per word; **higher = more fragmented** |
| **Parity** | Similar token counts across languages for the same meaning |
| **Structure** | Don’t shatter indent, newlines, URLs, digits |
| **Paired with the checkpoint** | Hard rule — see example |
| **Vocab size / speed / reversible / specials / downstream** | Bigger often shorter but fatter embeddings; must be fast and lossless; final judge is trained quality |

**“Must pair with the same weights” — and why you usually don’t swap cuts:**

Next chapter: each id looks up **one row** in the **embedding table** (part of the **weights**).  
Normal use is already: **however you split is how you look up** — one checkpoint ships with one tokenizer; you don’t casually change mid-flight.

So why stress mismatches? Because **many tables exist**; breakage is usually a **wrong pair**, not changing your mind mid-request:

| When tables differ | Example |
| --- | --- |
| **Different model families** | Llama ≠ Qwen ≠ GPT \`cl100k\` / \`o200k\` |
| **Same vendor, new generation** | GPT-3.5 ~100K → GPT-4o ~200K (retrain/expand) |
| **Train from scratch** | New BPE on your corpus; old table won’t fit new embeddings |
| **Added specials** | Finetune/tools append markers; merges often unchanged, id space grows |
| **Ops mistake** | Llama weights + someone else’s \`tokenizer.json\` |

So:

- **Using one model:** split and look up with its table; no random swaps.  
- **Why we teach pairing:** avoid “these weights + that family’s tokenizer”; also why vendors differ and generations expand vocabs.  

Example: train-time \`the\` → **65** → row 65 means “the”. Wrong table maps \`the\` → **902** → model still reads row 902 → nonsense.

\`\`\`mermaid
flowchart LR
  pair[OK: Llama weights + Llama tokenizer] --> ok[Ids match]
  mismatch[Bad: Llama weights + other tokenizer] --> bad[Ids shift → broken]
\`\`\`

Papers: compression/parity help, not enough alone.

### 6. What we do here

Official **Llama 3.2** BPE (\`128256\`) — not because we might swap tables, but because **loading these weights only works with this table**. The teaching panel is intuition only.

## Run

Accept \`meta-llama/Llama-3.2-1B\` on Hugging Face, then:

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
\`\`\`

Expect ids, round-trip decode, bos / eos / pad. Mirror: \`export HF_ENDPOINT=https://hf-mirror.com\`

## Fix

Gated download—open when stuck:

- No license / login → script hints  
- \`401\` → Agree, then login again  
- Slow → HF mirror  

## Next

**Embedding** (\`?c=embedding\`): how an id becomes that row of numbers.

`,
  },
  {
    id: 'embedding',
    ep: '03',
    title: 'Embedding',
    summary: 'Token ids become vectors—the first learnable lookup.',
    script: 'chapters/01-inference/ep03_embedding.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Idea

Token ids are still just integers. The model needs **numbers it can multiply**—a vector per id.

**Embedding** is a lookup table of shape \`(vocab_size, dim)\`: row \`i\` is the vector for token id \`i\`. One forward = gather those rows. No “understanding” happens here yet; it is the first learnable map from symbols to geometry.

Important for Llama 3.2–style models:

- We do **not** add a separate absolute position table on top of the embedding. **Position comes later via RoPE** on Q/K.
- Often the same table is **tied** with the final LM head (same weights for “id → vector” and “vector → vocab scores”).

The lab uses a tiny fake vocab so it runs without downloading weights; shapes match the real idea: \`(batch, seq) → (batch, seq, dim)\`.

## Run

\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`

Expect shapes for ids and output, plus a note that position is RoPE’s job.

## Next

**RMSNorm** (\`?c=rmsnorm\`): stabilize vectors before the heavy layers.

`,
    latex: 'X = \\mathrm{Embed}(t_{1:L}) \\in \\mathbb{R}^{L \\times d}',
  },
  {
    id: 'rmsnorm',
    ep: '04',
    title: 'RMSNorm',
    summary: 'Keep layer scales sane before attention and MLP.',
    script: 'chapters/01-inference/ep04_rmsnorm.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Idea

Deep stacks blow up or die if every layer sees wild scales. **Normalization** keeps vectors in a sane range.

**RMSNorm** (root-mean-square norm) is the modern default here:

- For each token vector, compute RMS over the last dimension  
- Divide by that RMS (plus a tiny \`ε\`)  
- Multiply by a learnable scale \`γ\`  
- **No mean subtraction** (unlike classic LayerNorm)—cheaper, still stable enough in practice  

We use it as **pre-norm**: once before attention, once before the MLP. That is what Llama / Qwen-style dense stacks do.

$$\\mathrm{RMSNorm}(x)=x/\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}\\odot\\gamma$$

## Run

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

Expect: same shape in/out; per-token RMS after norm ≈ 1 when \`γ\` starts as ones.

## Next

**RoPE** (\`?c=rope\`): bake position into Q/K with rotations.

`,
    latex:
      '\\mathrm{RMSNorm}(x) = \\frac{x}{\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}} \\odot \\gamma',
  },
  {
    id: 'rope',
    ep: '05',
    title: 'RoPE',
    summary: 'Put position into Q/K with rotations—not a PE table on embeds.',
    script: 'chapters/01-inference/ep05_rope.py',
    partId: 'trunk',
    estimatedMinutes: 15,
    demo: 'rope',
    markdown: `## Idea

Attention compares queries and keys. Without position, “A then B” and “B then A” look the same.

**RoPE** (Rotary Position Embedding) rotates each Q/K vector by an angle that depends on its **position**. Relative distance then shows up naturally in the dot product—no big learned position table glued onto embeddings.

Llama 3.2 uses a large base frequency (\`rope_theta\`) so long contexts stay usable; later we will see **wavelength scaling** for even longer prompts.

On this page: drag the interactive picture. In code: build cos/sin caches, \`rotate_half\`, then \`q' = q·cos + rotate(q)·sin\` (same for k).

## Run

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

Expect a short demo of applying RoPE and checking shapes.

## Next

**GQA** (\`?c=gqa\`): attention with fewer KV heads.

`,
    latex:
      'q_m^{\\prime} = R_{\\Theta,m} q_m,\\quad k_n^{\\prime} = R_{\\Theta,n} k_n',
  },
  {
    id: 'gqa',
    ep: '06',
    title: 'GQA attention',
    summary: 'Share KV across query groups to shrink the decode cache.',
    script: 'chapters/01-inference/ep06_gqa.py',
    partId: 'trunk',
    estimatedMinutes: 18,
    demo: 'gqa',
    markdown: `## Idea

Full multi-head attention: every query head has its own key/value head. That makes the **KV cache** (next chapters) large at decode time.

**GQA** (Grouped-Query Attention): fewer KV heads than query heads. Several query heads **share** one KV head. Llama 3.2 1B uses **32 query / 8 KV** heads—same quality ballpark, smaller cache.

Causal mask still applies: position \`t\` only sees positions \`≤ t\`.

Flow in our code: project Q/K/V → reshape heads → RoPE on Q/K → expand KV with \`repeat_interleave\` → scaled dot-product attention → output projection.

## Run

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

Expect a tiny forward that prints head counts and output shape. Use the interactive diagram on this page.

## Next

**SwiGLU** (\`?c=swiglu\`): the feed-forward half of each block.

`,
  },
  {
    id: 'swiglu',
    ep: '07',
    title: 'SwiGLU',
    summary: 'Modern feed-forward: gated SiLU times up, then down.',
    script: 'chapters/01-inference/ep07_swiglu.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Idea

After attention mixes tokens, each position still needs a **per-token MLP** to transform features.

Modern stacks use **SwiGLU** instead of a plain GELU MLP:

$$\\mathrm{FFN}(x)=W_d\\big(\\mathrm{SiLU}(W_g x)\\odot W_u x\\big)$$

Three bias-free linears: **gate**, **up**, **down**. SiLU on the gate, multiply by up, then down-project. Same width ideas as Llama / Qwen dense models.

## Run

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

Expect input/output same \`embed_dim\`, intermediate wider.

## Next

**Block** (\`?c=block\`): glue norm + attention + SwiGLU with residuals.

`,
    latex: '\\mathrm{FFN}(x)=W_d\\big(\\mathrm{SiLU}(W_g x)\\odot W_u x\\big)',
  },
  {
    id: 'block',
    ep: '08',
    title: 'Transformer block',
    summary: 'One residual sandwich you stack many times.',
    script: 'chapters/01-inference/ep08_block.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Idea

One **Transformer block** is the repeatable sandwich:

\`\`\`
x → RMSNorm → Attention → + residual
  → RMSNorm → SwiGLU   → + residual
\`\`\`

Same shape in and out, so you stack it \`N\` times (Llama 3.2 1B: **16** layers). Pre-norm (normalize *before* the sublayer) is what modern dense models use; residuals keep gradients alive.

Optional KV cache threads through attention for decode (you will use it soon).

## Run

\`\`\`bash
python chapters/01-inference/ep08_block.py
\`\`\`

Expect one block forward; shapes unchanged across the residual path.

## Next

**Stack + LM head** (\`?c=stack\`): embed → many blocks → vocab scores.

`,
  },
  {
    id: 'stack',
    ep: '09',
    title: 'Stack + LM head',
    summary: 'Full path: embed → blocks → vocab scores (logits).',
    script: 'chapters/01-inference/ep09_stack.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Idea

Full model path:

\`token ids → Embedding → Block×N → final RMSNorm → Linear(vocab)\`

The last linear is the **LM head**: each position gets a score per vocab entry (**logits**). Highest (or sampled) id = next token.

Llama 3.2 often **ties** LM head weights with the embedding table—one matrix, two roles.

## Run

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

Expect a tiny random model forward and logits shape \`(batch, seq, vocab)\`.

## Next

**Load weights** (\`?c=weights\`): swap random init for real Llama tensors.

`,
  },
  {
    id: 'weights',
    ep: '10',
    title: 'Load weights',
    summary: 'Copy real Llama 3.2 tensors into our modules.',
    script: 'chapters/01-inference/ep10_load_weights.py',
    partId: 'trunk',
    estimatedMinutes: 20,
    markdown: `## Idea

So far the graph was right but weights were random—output is nonsense.

**\`from_pretrained\`** builds our \`Llama\` modules, then copies tensors from the official Hugging Face Llama 3.2 checkpoint using a **key map** (HF names → arche names). After that, forward math matches the released model (up to numerics).

Needs accepting the model license and \`huggingface-cli login\` (same gate as the tokenizer).

## Run

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

Expect confirmation that weights landed (parameter counts / a short sanity print). Mirror tip: \`export HF_ENDPOINT=https://hf-mirror.com\`.

## Fix

Same gated-model issues as the tokenizer chapter: license, login, network.

## Next

**Generate** (\`?c=generate\`): turn logits into text, token by token.

`,
  },
  {
    id: 'generate',
    ep: '11',
    title: 'Text generation',
    summary: 'Loop: logits → sample → append → repeat.',
    script: 'chapters/01-inference/ep11_generate.py',
    partId: 'trunk',
    estimatedMinutes: 15,
    markdown: `## Idea

Generation is a loop, not one shot:

1. Run the model on the current ids → logits for the **last** position  
2. Optionally soften with **temperature**, keep only **top-k**  
3. **Greedy** (\`argmax\`) or **sample** from the distribution  
4. Append the new id; stop on EOS or max length  

With a KV cache (next chapter) step 1 becomes cheap after the first prefill. This chapter focuses on the sampling loop itself.

## Run

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

Expect a short decoded string (needs weights from the previous chapter for real text; otherwise follow the script’s notes).

## Next

**KV cache** (\`?c=kv-cache\`): stop recomputing the whole prompt every step.

`,
  },
  {
    id: 'kv-cache',
    ep: '12',
    title: 'KV cache',
    summary: 'Prefill once; decode only appends new K/V.',
    script: 'chapters/01-inference/ep12_kv_cache.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'kv',
    markdown: `## Idea

Naive generate re-runs attention over **all** past tokens every step. That repeats the same K/V work for the prompt.

**KV cache:** on the first pass (**prefill**), store every layer’s keys and values. Each new token (**decode**) only computes K/V for itself and **appends** to the cache, then attends over the full cached sequence.

Same answers (in exact arithmetic), far less work as the reply grows. See the interactive strip on this page.

## Run

\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

Expect a comparison or demo showing cached decode stepping.

## Next

**RoPE scaling** (\`?c=rope-scaling\`): keep RoPE sane for long prompts.

`,
  },
  {
    id: 'rope-scaling',
    ep: '13',
    title: 'RoPE scaling',
    summary: 'Stretch long RoPE wavelengths for longer prompts.',
    script: 'chapters/01-inference/ep13_rope_scaling.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## Idea

RoPE angles were tuned for a training context length. Push much longer and relative positions can “wrap” oddly.

**Llama 3.2-style RoPE scaling** keeps a large \`rope_theta\` and smoothly rescales **long wavelengths** using factor / low / high frequency bands (Hugging Face–compatible formula). Short-range behavior stays familiar; long-range gets stretched so longer prompts remain usable.

This chapter is about the **schedule**, not inventing a new attention.

## Run

\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

Expect prints that contrast unscaled vs scaled inverse frequencies / shapes.

## Next

**MoE** (\`?c=moe\`): sparse experts instead of one big MLP.

`,
  },
  {
    id: 'moe',
    ep: '14',
    title: 'MoE',
    summary: 'Many experts; each token runs only top-k.',
    script: 'chapters/01-inference/ep14_moe.py',
    partId: 'infer',
    estimatedMinutes: 15,
    markdown: `## Idea

A dense MLP runs the **same** feed-forward for every token. **Mixture of Experts (MoE)** keeps many expert MLPs (here: SwiGLU experts) and a **router** that, per token, picks only the **top-k** experts to run.

You can grow total parameters a lot while each token pays for only a few experts—**sparse compute** (Mixtral-style idea). This chapter is a teaching MoE block, not a full production Mixtral stack.

## Run

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

Expect router choices / shapes for a tiny MoE forward.

## Next

**MLA** (\`?c=mla\`): compress what you cache for decode.

`,
  },
  {
    id: 'mla',
    ep: '15',
    title: 'MLA',
    summary: 'Cache a small latent instead of full K/V.',
    script: 'chapters/01-inference/ep15_mla.py',
    partId: 'infer',
    estimatedMinutes: 18,
    markdown: `## Idea

GQA already shrinks KV heads. **MLA** (Multi-head Latent Attention, DeepSeek-style idea) goes further for decode memory: cache a **low-rank latent** (plus a small RoPE stream) instead of full per-head K and V, then reconstruct what attention needs on the fly.

Teaching goal: feel *why* latent KV helps memory—not ship a bit-identical DeepSeek checkpoint.

## Run

\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

Expect a compact forward that shows latent cache shapes.

## Next

**Device · INT8 · chat** (\`?c=device-quant-chat\`): where it runs, how weights shrink, how chats are formatted.

`,
  },
  {
    id: 'device-quant-chat',
    ep: '16',
    title: 'Device · INT8 · chat',
    summary: 'Pick device, shrink weights, format chat turns.',
    script: 'chapters/01-inference/ep16_device_quant_chat.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'chat',
    markdown: `## Idea

Three practical pieces that sit around the math core:

1. **Device** — pick CUDA if present, else Apple MPS, else CPU (\`pick_device\`)  
2. **INT8 (teaching)** — store weights in int8 to shrink memory; compute still follows a simple dequant path for learning  
3. **Chat template** — turn role messages (system/user/assistant) into the token string the model expects; if HF’s template is missing, we fall back to a plain readable format (try the demo on this page)

## Run

\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`

Expect device choice, a quant note, and a rendered chat string.

## Next

**Continuous batching** (\`?c=continuous-batching\`): many short generations sharing steps.

`,
  },
  {
    id: 'continuous-batching',
    ep: '17',
    title: 'Continuous batching',
    summary: 'Many short jobs share one stepping loop.',
    script: 'chapters/01-inference/ep17_continuous_batching.py',
    partId: 'infer',
    estimatedMinutes: 15,
    demo: 'batch',
    markdown: `## Idea

Serving many users naively = one generate loop each (GPUs idle between people).

**Continuous batching** (teaching scheduler here): keep several short decode jobs in one stepping loop; when a job hits max new tokens (or EOS), retire it and optionally admit another. Everyone shares the same “time step” rhythm.

This is **not** full paged attention / vLLM—just the scheduling shape. Try the demo on this page.

## Run

\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`

Expect several jobs advancing together until each finishes.

## Next

**Pretrain** (\`?c=pretrain\`): a tiny next-token training loop.

`,
  },
  {
    id: 'pretrain',
    ep: '18',
    title: 'Pretrain',
    summary: 'Toy next-token training—feel the API.',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'train',
    estimatedMinutes: 15,
    demo: 'pretrain',
    markdown: `## Idea

**Pretraining** = learn to predict the next token on lots of text. Loss is usually cross-entropy between logits and the true next id; an optimizer (here **AdamW**) updates weights.

Our loop is intentionally **tiny** (toy batch, few steps): you feel the API and watch loss move—not train a real 1B on the internet. On this page, drag learning rate / steps and run a real tiny loop in the browser.

## Run

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

Expect a few steps of decreasing-ish toy loss. The page demo simulates the same idea visually.

## Next

**SFT · DPO** (\`?c=sft-dpo\`): follow instructions and preferences.

`,
  },
  {
    id: 'sft-dpo',
    ep: '19',
    title: 'SFT · DPO',
    summary: 'Mask prompts for SFT; one preference step for DPO.',
    script: 'chapters/02-training/ep19_sft_dpo.py',
    partId: 'train',
    estimatedMinutes: 18,
    demo: 'train',
    markdown: `## Idea

After pretrain, models still need to **follow chats** and **prefer better answers**.

- **SFT** (supervised fine-tune): train on prompt→response examples, but **mask the prompt** so loss only hits the reply tokens.  
- **DPO** (Direct Preference Optimization): given chosen vs rejected replies, push the model toward chosen **relative to a frozen reference**—no separate reward model in the basic form. We run **one teaching step**.

## Run

\`\`\`bash
python chapters/02-training/ep19_sft_dpo.py
\`\`\`

Expect SFT masking behavior and a single DPO update printout.

## Next

**Frontier checkpoints** (\`?c=frontier-ckpt\`): save/load MoE & MLA shaped weights.

`,
  },
  {
    id: 'frontier-ckpt',
    ep: '20',
    title: 'Frontier checkpoints',
    summary: 'Save/load MoE & MLA; map Mixtral expert keys.',
    script: 'chapters/02-training/ep20_frontier_ckpt.py',
    partId: 'frontier',
    estimatedMinutes: 15,
    markdown: `## Idea

Frontier modules (MoE, MLA) need the same discipline as dense: **save and load** without silent shape bugs.

This chapter round-trips teaching MoE/MLA checkpoints and shows how **Mixtral-style** expert keys (\`w1/w2/w3\`) map into our module names—so reading real open checkpoints is less mysterious.

## Run

\`\`\`bash
python chapters/02-training/ep20_frontier_ckpt.py
\`\`\`

Expect successful save→load and a key-map demonstration.

## Next

**HF align** (\`?c=hf-align\`): prove dense logits match Hugging Face when weights exist.

`,
  },
  {
    id: 'hf-align',
    ep: '21',
    title: 'HF logits alignment',
    summary: 'Check our logits against Hugging Face on a short prompt.',
    script: 'chapters/01-inference/ep21_hf_align.py',
    partId: 'frontier',
    estimatedMinutes: 20,
    markdown: `## Idea

Trust but verify: after \`from_pretrained\`, run the **same short prompt** through arche and Hugging Face Llama 3.2; compare **argmax** and **max absolute logit difference**.

If gated weights are missing, the chapter exits cleanly (CI still has a synthetic mapper test). This is your “did we wire the dense trunk correctly?” gate.

## Run

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

Expect a small diff report when weights are available.

## Fix

No weights / no login → clean skip or Fix text from the script; not a logic failure of arche itself.

## Next

**Sharded MoE/MLA** (\`?c=sharded-frontier\`): how big checkpoints are split on disk.

`,
  },
  {
    id: 'sharded-frontier',
    ep: '22',
    title: 'Sharded MoE / MLA',
    summary: 'index.json + shards → reload MoE/MLA.',
    script: 'chapters/02-training/ep22_sharded_frontier.py',
    partId: 'frontier',
    estimatedMinutes: 15,
    markdown: `## Idea

Huge models rarely ship as one file. Hugging Face-style layouts use an **\`index.json\`** plus many **\`.safetensors\` / \`.pt\` shards**.

We build a **synthetic** index + shards for Mixtral-shaped MoE and DeepSeek-shaped MLA, reload them, and \`allclose\`—so you see the packing/unpacking pattern without downloading tens of GB.

## Run

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`

Expect shard write/read and matching tensors.

## Next

**Paged KV · CUDA graphs** (\`?c=paged-cuda-graph\`): serving-memory and capture ideas.

`,
  },
  {
    id: 'paged-cuda-graph',
    ep: '23',
    title: 'Paged KV · CUDA graphs',
    summary: 'Block-table KV pool and fixed-shape CUDA graphs.',
    script: 'chapters/01-inference/ep23_paged_cuda_graph.py',
    partId: 'frontier',
    estimatedMinutes: 18,
    markdown: `## Idea

Two serving ideas, teaching-sized:

1. **Paged KV** — store KV in fixed-size **blocks** + a **block table**, gather what attention needs (vLLM-inspired shape). Stops one long sequence from needing one giant contiguous cache allocation story.  
2. **CUDA Graph** — when CUDA exists, **capture** a fixed-shape compute graph to cut launch overhead; decode may stay eager until KV writes are safely in-place.

Labs show the mechanisms; they are not a drop-in production engine.

## Run

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

Expect paged gather behavior; graph capture only if CUDA is available.

## Next

Path complete. Revisit weak chapters, or run HF align with real weights when ready.

`,
  },
];
