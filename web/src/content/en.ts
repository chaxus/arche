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

### 1. Why this specimen

We need one small, open, production-like model to hand-write end to end.

### 2. What “the model” is here

A **Llama 3.2–style** next-token stack: tokenizer → embed → blocks → LM head. People call this a **dense decoder**—learn the name as you go.

\`\`\`mermaid
flowchart LR
  tok[Tokenizer] --> emb[Embedding]
  emb --> stack[Blocks]
  stack --> head[LM head]
\`\`\`

### 3–5. Why Llama 3.2

~1B, downloadable weights, same bricks as many production dense models (RMSNorm, RoPE, SwiGLU, GQA). Later parts add inference, tiny train, and frontier primers.

### 6. What we do

Print config sizes in the lab—not chat yet.

## Run

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

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
    summary: 'Ids are labels; embedding gathers a row of floats you can compute with.',
    script: 'chapters/01-inference/ep03_embedding.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    demo: 'embedding',
    markdown: `## Idea

### 1. Why embedding

The last chapter left you with **token ids**, e.g. \`[3, 7, 11]\`.

Ids are only labels—“cabinet 3”, “cabinet 7”. The network multiplies and attends; it needs **lists of floats**, not cabinet numbers.

**Embedding** assigns every id a fixed-length list of numbers (a **vector**). From here on, the stack eats those vectors.

One line: **tokenizer gives ids; embedding turns ids into rows you can compute with.**

### 2. What it does (lookup)

Think of a spreadsheet:

| id (row) | dim 0 | dim 1 | dim 2 | … |
| --- | --- | --- | --- | --- |
| 0 | 0.12 | -0.04 | 0.31 | … |
| 1 | -0.22 | 0.08 | 0.01 | … |
| 3 | **0.55** | **-0.17** | **0.09** | … |
| 7 | **0.03** | **0.41** | **-0.28** | … |
| … | … | … | … | … |

- **vocab size** rows (one per possible id)  
- **dim** floats per row (2048 for Llama 3.2 1B)  
- Forward pass is mostly **gather rows by id**—not “understanding” yet

Example: ids \`[3, 7, 11]\` → shape \`(1, 3, dim)\` = rows 3, 7, and 11 stacked.

\`\`\`mermaid
flowchart LR
  ids["ids: 3, 7, 11"] --> pick[gather rows]
  table[Embedding table] --> pick
  pick --> out["three vectors → (1,3,dim)"]
\`\`\`

Try the **tiny table** above the lesson (click ids, see which rows light up). The script below uses the same idea with a fake vocab so you need no download.

### 3. How this ties to the tokenizer and weights

| Idea | Plain meaning |
| --- | --- |
| **Tokenizer table** | text ↔ id (previous chapter) |
| **Embedding table** | id ↔ that row of floats |
| **Weights** | those floats are learned and stored in the checkpoint |

So **one tokenizer + matching embedding** must stay paired.  
If training saw \`the\` → id \`65\`, row 65 became “the”. A foreign tokenizer that maps \`the\` → \`902\` still reads row 902 → garbage (the pairing story from the tokenizer lesson lands here).

Fresh or demo tables look almost random—**no semantics yet**. Meaning is trained later; embedding is only the first learnable map from symbols into geometry.

### 4. Design choices you will hear

| Choice | Plain idea | Today |
| --- | --- | --- |
| Absolute PE added on embeds | Learn a vector per position; add it on | Older stacks; **not** the Llama-style main path |
| **No PE here; RoPE later** | Position enters as rotations on Q/K | **Mainstream dense decoders** |
| **Tied LM head** | Reuse the same matrix for id→vec and vec→vocab scores | Common on small Llamas |

Easy mix-up: **position is not this chapter’s job**. The table says “which token”; “which place in the sequence” waits for **RoPE**.

### 5. What labs ship

Llama 3.2 / Qwen-style dense:

- Embedding = **lookup** (\`nn.Embedding\` or equivalent)  
- **No** absolute PE added on the embed  
- Small models often **tie** embed ↔ LM head  

Vendors differ on vocab size, dim, tying; “id → take a row” is the same idea.

### 6. What “good” means · what we do

| Check | Plain meaning |
| --- | --- |
| Shape | \`(B, T)\` ids → \`(B, T, dim)\` vectors |
| Pairing | row \`i\` ↔ tokenizer id \`i\` |
| Position | If the stack assumes RoPE, don’t sneak absolute PE back in here |

\`TokenEmbedding\` in \`arche/\`. The lab uses a tiny fake vocab (e.g. 32 rows)—**no real weight download**; real Llama vocab is ~128K rows with the same logic. When you run the script, check id/output shapes and the note that position is RoPE’s job.

## Run

\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`

Expect real vs demo vocab size, \`embed_dim\`, and id/output shapes.

## Next

**RMSNorm** (\`?c=rmsnorm\`): stabilize scales before the heavy layers.

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

### 1. Why normalize

Deep stacks blow up or die if every layer sees wild scales. Normalization keeps vectors usable.

### 2. What RMSNorm does

For each token vector (last dim):

1. Compute RMS  
2. Divide by RMS (+ tiny \`ε\`)  
3. Multiply by learnable \`γ\`  
4. **No mean subtract** (unlike LayerNorm)

We use **pre-norm**: before attention and before MLP.

$$\\mathrm{RMSNorm}(x)=x/\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}\\odot\\gamma$$

### 3. Names you will hear

| Name | Plain idea | Today |
| --- | --- | --- |
| LayerNorm | Mean-center + scale | Older GPT-2-style stories |
| **RMSNorm** | Scale by RMS only | **Llama / Qwen default** |
| Pre-norm vs post-norm | Norm before vs after sublayer | Modern stacks: pre-norm |

### 4. What labs ship

Llama 3.2, Qwen dense, most new open decoders: RMSNorm + pre-norm.

### 5. What “good” means

- Same shape in/out  
- With \`γ=1\`, post-norm RMS ≈ 1  
- Stable training without being the whole story (optimizer/data still matter)  

### 6. What we do

\`RMSNorm\` in \`arche/norms.py\`. Lab prints shapes and per-token RMS.

## Run

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## Next

**RoPE** (\`?c=rope\`): bake position into Q/K.

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

### 1. Why position

Without position, “A then B” and “B then A” look the same to attention.

### 2. What RoPE does

**RoPE** rotates each Q/K by an angle from its **position**. Relative distance shows up in the dot product—no big PE table on embeddings.

On this page: drag the interactive picture. In code: cos/sin cache, \`rotate_half\`, then \`q' = q·cos + rotate(q)·sin\`.

### 3. Alternatives

| Approach | Plain idea | Today |
| --- | --- | --- |
| Absolute learned PE | Add a vector per index | Older stacks |
| Sinusoidal PE | Fixed sin/cos by position | Transformer original |
| **RoPE** | Rotate Q/K | **Mainstream decoders** |
| ALiBi / other biases | Distance bias in scores | Some long-context lines |

### 4. What labs ship

Llama 3 / 3.2: large \`rope_theta\`; later chapters add **wavelength scaling** for longer context.

### 5. What “good” means

- Relative positions matter in attention  
- Shapes of Q/K unchanged  
- Long context may need scaling (next specialty chapter)  

### 6. What we do

\`arche/rope.py\`. Lab applies RoPE and checks shapes.

## Run

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## Next

**GQA** (\`?c=gqa\`): fewer KV heads.

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

### 1. Why GQA

Full MHA: every query head has its own KV head → **KV cache** grows fast at decode.

### 2. What GQA does

**Grouped-Query Attention**: fewer KV heads than query heads; several Q heads **share** one KV. Still **causal** (position \`t\` sees \`≤ t\`).

Flow: project → reshape → RoPE → expand KV → SDPA → out proj. See the diagram on this page.

### 3. Family of attention layouts

| Layout | Plain idea | Today |
| --- | --- | --- |
| MHA | Q heads = KV heads | Classic |
| MQA | One KV for all Q | Extreme save |
| **GQA** | KV groups shared by Q | **Llama 3.2 1B: 32Q / 8KV** |
| MLA | Compress KV to a latent | DeepSeek-style (later chapter) |

### 4. What labs ship

Llama 3.2 1B uses GQA. Many open 7B–70B models use GQA for decode memory.

### 5. What “good” means

- Correct \`n_rep = n_q / n_kv\`  
- Causal mask / \`is_causal\` right for prefill vs decode  
- Cache smaller than full MHA without collapsing quality entirely  

### 6. What we do

\`CausalSelfAttention\` in \`arche/attention.py\`. Lab prints head counts and shapes.

## Run

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## Next

**SwiGLU** (\`?c=swiglu\`): the feed-forward half.

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

### 1. Why an MLP after attention

Attention mixes tokens; each position still needs a **per-token** transform.

### 2. What SwiGLU does

$$\\mathrm{FFN}(x)=W_d\\big(\\mathrm{SiLU}(W_g x)\\odot W_u x\\big)$$

Three bias-free linears: **gate**, **up**, **down**.

### 3. Alternatives

| FFN | Plain idea | Today |
| --- | --- | --- |
| GELU MLP (GPT-2 style) | One hidden width + GELU | Older teaching fossils |
| **SwiGLU** | Gated SiLU × up | **Llama / Qwen dense** |
| MoE of SwiGLU experts | Router picks top-k | Mixtral-style (later) |

### 4. What labs ship

Modern dense decoders: SwiGLU (or close gated variants).

### 5. What “good” means

- In/out \`embed_dim\`; intermediate wider  
- No bias on the three linears (Llama-style)  

### 6. What we do

\`SwiGLU\` in \`arche/mlp.py\`.

## Run

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## Next

**Block** (\`?c=block\`): glue the sandwich.

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

### 1. Why a block

One reusable unit you can stack \`N\` times.

### 2. What it is

\`\`\`
x → RMSNorm → Attn → + residual
  → RMSNorm → SwiGLU → + residual
\`\`\`

Same shape in/out. Optional KV cache through attention.

### 3. Layout choices

| Choice | Plain idea | Today |
| --- | --- | --- |
| Post-norm | Norm after residual | Older |
| **Pre-norm** | Norm before sublayer | **Modern default** |
| Parallel attn∥MLP | Some non-Llama layouts | Not our trunk |

### 4. What labs ship

Llama 3.2 1B: **16** such blocks.

### 5. What “good” means

- Residual paths keep shapes  
- Cache threading doesn’t drop layers  

### 6. What we do

\`TransformerBlock\` in \`arche/block.py\`.

## Run

\`\`\`bash
python chapters/01-inference/ep08_block.py
\`\`\`

## Next

**Stack + LM head** (\`?c=stack\`).

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

### 1. Why stack

One block isn’t a language model—many blocks + a vocab head are.

### 2. Full path

\`ids → Embed → Block×N → final RMSNorm → Linear(vocab)\`

Last scores = **logits** for the next token. Often **tie** LM head ↔ embedding.

### 3. Pieces

| Piece | Role |
| --- | --- |
| Embedding | ids → vectors |
| Blocks | contextualize |
| Final norm | stabilize before head |
| LM head | vectors → vocab scores |

### 4. What labs ship

Llama-style dense decoder end-to-end (still random weights until the load chapter).

### 5. What “good” means

- Logits shape \`(B, T, vocab)\`  
- Tie setting matches the config you claim  

### 6. What we do

\`Llama\` in \`arche/model.py\`.

## Run

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

## Next

**Load weights** (\`?c=weights\`).

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

### 1. Why load weights

Right graph + random weights = nonsense text.

### 2. What we do

\`from_pretrained\` builds our modules, then copies HF Llama 3.2 tensors via a **key map**.

### 3. Pitfalls

| Issue | Meaning |
| --- | --- |
| Wrong tokenizer | Ids won’t match rows (see tokenizer chapter) |
| Missing gated access | Can’t download |
| Key map bugs | Silent wrong layers |

### 4. What labs ship

Official HF Llama 3.2 checkpoints (license + login).

### 5. What “good” means

- Param counts look right  
- Later HF align chapter: logits close on a short prompt  

### 6. What we do

\`arche/load_weights.py\` + chapter script.

## Run

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

## Fix

Same gated issues as tokenizer: license, login, mirror.

## Next

**Generate** (\`?c=generate\`).

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

### 1. Why a loop

One forward gives scores for the next piece—not a full reply.

### 2. What generation does

1. Logits at the **last** position  
2. Optional temperature / top-k  
3. Greedy or sample  
4. Append; stop on EOS or length  

KV cache (next chapter) makes step 1 cheap after prefill.

### 3. Knobs

| Knob | Effect |
| --- | --- |
| temperature → 0 | Greedy / sharp |
| higher temperature | More random |
| top-k | Keep only top candidates |

### 4. What labs ship

Same loop idea in every chat stack; sampling details vary.

### 5. What “good” means

- Stops cleanly  
- Respects EOS when provided  
- With real weights, text is coherent (not this chapter’s only goal)  

### 6. What we do

\`generate\` in \`arche/generate.py\`.

## Run

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

## Next

**KV cache** (\`?c=kv-cache\`).

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

### 1. Why cache

Naive decode recomputes K/V for the whole prompt every step.

### 2. What KV cache does

**Prefill** once → store per-layer K/V. Each **decode** step appends one new K/V and attends over the cache. See the strip on this page.

### 3. Related ideas

| Idea | Role |
| --- | --- |
| Contiguous cache | Simple \`cat\` along time (our teaching cache) |
| Paged KV | Block table + gather (later primer) |
| GQA/MLA | Shrink what you store |

### 4. What labs ship

Every production decoder uses some KV cache; engines differ on paging/layout.

### 5. What “good” means

- Same answers as full recompute (up to numerics)  
- Work grows with new tokens, not full prompt length each step  

### 6. What we do

\`KVCache\` in \`arche/cache.py\`.

## Run

\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

## Next

**RoPE scaling** (\`?c=rope-scaling\`).

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

### 1. Why scale RoPE

Angles were tuned for a training length; much longer prompts can behave oddly.

### 2. What Llama-style scaling does

Keep large \`rope_theta\`; smoothly rescale **long wavelengths** (factor / low / high)—HF-compatible.

### 3. Landscape

| Method | Plain idea |
| --- | --- |
| NTK / YaRN / … | Other long-context RoPE recipes |
| **Llama 3 wavelength smoothing** | What we mirror here |
| Bigger context train | Data + length, not only formulas |

### 4. What labs ship

Llama 3.2 long-context schedule in the official configs.

### 5. What “good” means

- Short-range stays familiar  
- Long-range stretches instead of wrapping badly  

### 6. What we do

Scaling helpers in \`arche/rope.py\` + chapter script.

## Run

\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

## Next

**MoE** (\`?c=moe\`).

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

### 1. Why MoE

One dense MLP per token is simple but capacity costs compute for everyone.

### 2. What MoE does

Many expert MLPs (here SwiGLU) + a **router**: each token runs only **top-k** experts → sparse compute.

### 3. Landscape

| Idea | Note |
| --- | --- |
| Dense FFN | Always-on |
| **MoE (Mixtral-style)** | Top-k experts |
| Shared experts / deep variants | Product-specific |

### 4. What labs ship

Teaching MoE block—not a full Mixtral production stack.

### 5. What “good” means

- Router picks valid experts  
- Load isn’t insanely skewed (real training cares; we only taste the API)  

### 6. What we do

\`arche/moe.py\` primer.

## Run

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

## Next

**MLA** (\`?c=mla\`).

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

### 1. Why MLA

GQA shrinks KV heads; decode memory can still dominate.

### 2. What MLA does

Cache a **low-rank latent** (+ small RoPE stream) instead of full per-head K/V; reconstruct what attention needs.

### 3. Landscape

| Idea | Role |
| --- | --- |
| GQA | Fewer KV heads |
| **MLA** | Latent KV (DeepSeek-style idea) |
| Paged KV | How storage is allocated |

### 4. What labs ship

Teaching shapes—not bit-identical DeepSeek weights.

### 5. What “good” means

- Latent cache smaller than full KV  
- Forward still runs  

### 6. What we do

\`arche/mla.py\` primer.

## Run

\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

## Next

**Device · INT8 · chat** (\`?c=device-quant-chat\`).

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

### 1. Why these three

Math core isn’t enough to *run* a chat: device, size, and message format matter.

### 2. What we cover

1. **Device** — CUDA → MPS → CPU  
2. **INT8 (teaching)** — smaller weights; simple dequant path  
3. **Chat template** — roles → string the model expects (HF or plain fallback; try the demo)

### 3. Landscape

| Topic | Production note |
| --- | --- |
| Quant | INT8/INT4/FP8… many recipes |
| Templates | Model-specific; don’t invent casually |
| Devices | Same graph, different backends |

### 4–6. What we do

Teaching helpers + chapter script—not a full quant product.

## Run

\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`

## Next

**Continuous batching** (\`?c=continuous-batching\`).

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

### 1. Why

One generate loop per user wastes GPU when people finish at different times.

### 2. What continuous batching is here

Several short decode jobs share one stepping loop; finished jobs retire. **Not** full paged attention / vLLM. Try the demo.

### 3. Landscape

| Layer | Role |
| --- | --- |
| Scheduling | Who is active this step |
| Paging / kernels | How memory & compute run (later primer) |

### 4–6. What we do

Teaching scheduler shape in the chapter script.

## Run

\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`

## Next

**Pretrain** (\`?c=pretrain\`).

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

### 1. Why pretrain

Learn next-token prediction on lots of text before following chats.

### 2. What we run

Toy AdamW loop: logits vs next id, cross-entropy, step. **On this page:** drag lr / steps and train a tiny model in-browser.

### 3. Landscape

| Scale | Meaning |
| --- | --- |
| This lab | Feel the API |
| Real pretrain | Huge data, many GPUs, weeks |

### 4–6. What we do

\`arche/train/pretrain.py\` + page playground. Not internet-scale training.

## Run

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## Next

**SFT · DPO** (\`?c=sft-dpo\`).

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

### 1. Why

After pretrain, models still need to follow chats and prefer better answers.

### 2. What

- **SFT:** prompt→response; **mask the prompt**; loss on reply only  
- **DPO:** prefer chosen over rejected vs a frozen reference (one teaching step)

### 3. Landscape

RLHF / RLAIF / DPO / … many alignment recipes; we taste two APIs.

### 4–6. What we do

\`arche/train/sft.py\` (+ DPO helper) and the chapter script.

## Run

\`\`\`bash
python chapters/02-training/ep19_sft_dpo.py
\`\`\`

## Next

**Frontier checkpoints** (\`?c=frontier-ckpt\`).

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

### 1. Why

MoE/MLA modules still need save/load without silent shape bugs.

### 2. What

Round-trip teaching checkpoints; map Mixtral-style \`w1/w2/w3\` expert keys into our names.

### 3–6. What we do

Primer for reading open MoE checkpoints—not a full trainer.

## Run

\`\`\`bash
python chapters/02-training/ep20_frontier_ckpt.py
\`\`\`

## Next

**HF align** (\`?c=hf-align\`).

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

### 1. Why

Trust but verify the dense trunk wiring.

### 2. What

Same short prompt through arche vs HF Llama 3.2; compare argmax and max-abs logit diff. No weights → clean exit.

### 3–6. What we do

Chapter script + CI synthetic mapper test.

## Run

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

## Fix

Missing weights/login → skip/hints, not necessarily a logic bug.

## Next

**Sharded MoE/MLA** (\`?c=sharded-frontier\`).

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

### 1. Why

Huge models ship as \`index.json\` + many shard files.

### 2. What

Synthetic index + shards for Mixtral-shaped MoE / DeepSeek-shaped MLA; reload and \`allclose\`.

### 3–6. What we do

Packing/unpacking pattern without downloading tens of GB.

## Run

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`

## Next

**Paged KV · CUDA graphs** (\`?c=paged-cuda-graph\`).

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

### 1. Why

Serving needs smarter memory and lower launch overhead than a toy loop.

### 2. What

1. **Paged KV** — fixed-size blocks + block table + gather  
2. **CUDA Graph** — capture fixed-shape compute when CUDA exists  

Teaching mechanisms—not a drop-in vLLM.

### 3–6. What we do

\`arche/paged_kv.py\` (+ graph helper) and the chapter script.

## Run

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

## Next

Path complete—revisit weak chapters or run HF align with real weights.

`,
  },
];
