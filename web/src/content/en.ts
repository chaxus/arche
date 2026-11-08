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

### 1. What it is for

This chapter picks the specimen: a **Llama 3.2-style** next-token stack—tokenize → embed → block stack → LM head—often called a **dense decoder**. Every later chapter hangs on this trunk.

### 2. Why the course needs it

You need one small, open, production-like model to hand-write end to end. Without a shared specimen, RMSNorm / RoPE / GQA won’t share one config.

### 3. Where it comes from · how we chose

| Factor | Llama 3.2 1B |
| --- | --- |
| Size | ~1B, laptop-reachable |
| Open | Weights available with license |
| Parts | RMSNorm, RoPE, SwiGLU, GQA—like many production dense stacks |
| Path | Trunk → inference → tiny train → frontier primers |

### 4. Standards · what “good choice” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Config prints and is consistent | Vocab, depth, dim match docs |
| **Soft** | Carries the whole course | Later chapters plug in |
| **Soft** | Honest scope | Dense trunk + primers, not full vLLM / full pretrain |

### 5. What we do

Lab prints config sizes—not chat yet.

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

### 1. What the embedding table is for

The embedding table is a matrix of shape \`(vocab_size, dim)\`.  
**Row \`i\` = the float row for token id \`i\`.**

One hard job, plus two related roles:

| Role | Plain meaning |
| --- | --- |
| **① Interface (required)** | Turn discrete ids into floats later layers can multiply/add. Without this, attention / MLP have no valid input |
| **② Learnable identity** | Each id owns a row; training rewrites those floats so frequent / related tokens become useful |
| **③ Often doubles as output scoring (optional)** | Small Llamas tie the same matrix to the LM head: gather in, score the vocab out |

It is **not** a dictionary gloss and **not** full-sentence understanding. At inference it is mostly: **gather row by id**.  
(Locker intuition: number = id, slip = the row; try the table above this lesson.)

\`\`\`mermaid
flowchart LR
  text[text] --> tok[tokenize → ids]
  tok --> emb[embed: gather rows]
  emb --> rest[later layers see only floats]
\`\`\`

### 2. Why the LLM pipeline needs it

The stack is a chain of **tensor** ops (matmuls, norms, attention…).  
Tokenization only yields integer ids. As category indices they are fine; as semantic numbers they are not—adding \`id=3\` and \`id=7\` has no linguistic meaning.

So you need a fixed, learnable translation:

- Input: id → float row (embedding)  
- Middle: compute only on floats  
- Output: scores over the vocab → new ids → tokenizer decodes to text  

Without an embedding table you’d invent another “id → first-layer input” scheme; modern decoders almost always choose **lookup**: simple, trainable, one-to-one with the vocab.

Vs the tokenizer table: tokenizer answers “text → ids”; embedding answers “ids → the computable world.” Both are required and must stay paired.

### 3. Where the embedding table comes from

**Not** hand-written floats per word, and **not** another BPE run.

| Stage | What happens |
| --- | --- |
| **Spec** | rows = tokenizer vocab size; cols = model dim (hyperparameter, e.g. 2048) |
| **Init** | allocate the table; fill with random floats (nearly meaningless) |
| **Pretrain** | next-token loss; backprop **updates the embedding with the rest of the net** |
| **Save** | write the checkpoint (\`.safetensors\`, etc.) beside attention, MLP, … |
| **You use it** | download the official model → load the trained table; inference gathers only |
| **Fine-tune (optional)** | keep updating floats, or append rows (new specials) and train more |

This lesson’s script builds a tiny **random toy table** so you see gather. Real Llama’s embedding lives in the official weights, factory-paired with its tokenizer.

### 4. What standards exist? What counts as a “good” table?

Split **hard constraints** (must not break) from **quality** (shows up after training).

**Hard constraints**

| Check | Plain meaning |
| --- | --- |
| **Shape** | rows = vocab; cols = \`dim\`; \`T\` ids → \`T\` rows |
| **Tokenizer pairing** | row \`i\` always matches tokenizer id \`i\` |
| **Usable numbers** | no NaN/Inf; scales don’t explode (RMSNorm etc. help) |
| **Stack contract** | if the stack assumes RoPE, don’t sneak absolute PE onto embeds unless you change the whole stack |

**“Good” — unlike the tokenizer table**

Tokenizers can be judged alone (fertility, parity, …).  
Embedding floats are **learned with the whole net**; labs rarely publish a fertility-style score for “the embedding alone.” People usually look at:

| Signal | Plain meaning |
| --- | --- |
| **Downstream / pretrain metrics** | better LM loss / evals under the same architecture — the main judge |
| **Dim budget** | too small → weak; too big → more params, hungrier for data. Set by architecture/compute, not “bigger is better” |
| **Rare tokens** | seldom-updated rows may stay undertrained; worse with huge vocabs |
| **Neighbor sanity (optional)** | nearest rows sometimes look like related pieces; a checkup, not the only score |
| **Tying** | small models often benefit from tying to the LM head; larger ones sometimes untie |

Bottom line:

- **Valid**: right shape, paired tokenizer, stable train/infer.  
- **Good**: the model scores well—embedding is one learnable part, **not a separate gold medal**.  
- Shipping a model: use the official embedding + official tokenizer; don’t bolt on a “better” foreign table.

### 5. What labs ship · what we do

Llama 3.2 / Qwen: lookup embed; position via RoPE; small models often tie the LM head.

\`TokenEmbedding\` in \`arche/\`: toy table for gather; real Llama ~128K rows, same idea.

## Run

\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`

Expect vocab size, dim, ids → output shape, and a note that position is not done here.

## Next

**RMSNorm** (\`?c=rmsnorm\`): steady scales before heavy layers.

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

### 1. What it is for

**RMSNorm** rescales each token’s float row: compute RMS, divide by it, multiply by a learnable \`γ\`. Unlike LayerNorm, it **does not subtract the mean**.

In Llama-style stacks it runs before attention and before the MLP (**pre-norm**).

### 2. Why the LLM pipeline needs it

Deep stacks blow up or vanish without scale control. Norm is the “re-zero the ruler” step before the next heavy op. Without it, attention/MLP are hard to stack stably.

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Spec** | Normalize the last dim; \`γ\` length = model dim |
| **Init** | \`γ\` often starts at 1 |
| **Train** | Gradients update \`γ\` with the rest of the net |
| **Infer** | Apply the formula; shape unchanged |

$$\\mathrm{RMSNorm}(x)=x/\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}\\odot\\gamma$$

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Same shape in/out | Don’t change T or dim |
| **Hard** | \`ε\` present | Avoid div-by-zero |
| **Hard** | Agreed placement | This course: pre-norm |
| **Soft** | With \`γ=1\`, per-token RMS ≈ 1 | Sanity that the formula is right |
| **Soft** | Stabler training | Final judge is full-model loss |

### 5. What labs ship · what we do

Llama 3.2 / Qwen dense: RMSNorm + pre-norm.  
\`RMSNorm\` in \`arche/norms.py\`; lab prints shapes and per-token RMS.

## Run

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## Next

**RoPE** (\`?c=rope\`): write position into Q/K.

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

### 1. What it is for

**RoPE** rotates Q and K by an angle from each **position**. Relative distance shows up in the attention dot product—no big PE table on embeddings.

### 2. Why the LLM pipeline needs it

Self-attention alone is order-blind. Language is not. Embedding answers “which token”; RoPE answers “which place.” Modern dense decoders write position into Q/K.

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Spec** | Choose \`rope_theta\`; cache cos/sin per position |
| **Forward** | \`rotate_half\` on Q/K, combine with cos/sin |
| **Train/infer** | Same rotation; longer context may need scaling (later chapter) |

Drag the page viz to see angles change with position.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Q/K shape unchanged | Phase only |
| **Hard** | Relative position enters scores | Swap positions → scores change |
| **Hard** | Match checkpoint contract | \`theta\` / scaling must match weights |
| **Soft** | Stable inside train length | Full-model quality is the judge |
| **Soft** | Extra-long context | See RoPE scaling chapter |

### 5. What labs ship · what we do

Llama 3 / 3.2: RoPE + large \`rope_theta\`.  
\`arche/rope.py\`; lab applies RoPE and checks shapes.

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

### 1. What it is for

**GQA** lets groups of Q heads **share** fewer KV heads. Still causal (\`t\` sees only \`≤ t\`), but decode stores less K/V.

Flow: project → split heads → RoPE → expand KV → SDPA → out proj. See the page viz.

### 2. Why the LLM pipeline needs it

Full MHA: every Q head owns KV → **KV cache** grows fast. GQA trades a bit of flexibility for much smaller cache vs MHA; usually gentler than extreme MQA.

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Spec** | Choose \`n_q\`, \`n_kv\` (divides evenly); \`n_rep = n_q / n_kv\` |
| **Impl** | Broadcast/repeat KV per group, then attend |
| **Train** | Learn projections with the net |
| **Infer** | Same layout + cache for prefill/decode |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | \`n_q\` divisible by \`n_kv\` | Valid groups |
| **Hard** | Causal mask correct | No future tokens |
| **Hard** | Head counts match weights | Else shape/silent errors |
| **Soft** | Cache clearly smaller than MHA | The point of GQA |
| **Soft** | Quality doesn’t collapse vs MHA | Judge with evals/loss |

### 5. What labs ship · what we do

Llama 3.2 1B: **32 Q / 8 KV**.  
\`CausalSelfAttention\` in \`arche/attention.py\`; lab prints heads and shapes.

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

### 1. What it is for

Attention mixes **across positions**; each position still needs its own nonlinear map—the **FFN / MLP**.  
Llama-style **SwiGLU**: three bias-free linears (gate / up / down), gated product, project back.

$$\\\\mathrm{FFN}(x)=W_d\\\\big(\\\\mathrm{SiLU}(W_g x)\\\\odot W_u x\\\\big)$$

### 2. Why the LLM pipeline needs it

Attention mostly “reads others.” The model also needs rich per-token transforms. SwiGLU is the common gated FFN in modern dense decoders (vs older GELU MLPs).

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Spec** | Wider hidden (\`ffn_dim\`); three bias-free mats |
| **Init / train** | With the rest of the net |
| **Variant** | MoE experts are often SwiGLUs (later) |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | In/out \`embed_dim\` | Width only in the middle |
| **Hard** | No bias (Llama contract) | Match weight layout |
| **Soft** | Gains vs GELU MLP | Full-model metrics |

### 5. What labs ship · what we do

Modern dense: SwiGLU or close gated cousins.  
\`SwiGLU\` in \`arche/mlp.py\`.

## Run

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## Next

**Transformer block** (\`?c=block\`): the sandwich.

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

### 1. What it is for

A **Transformer block** is one repeatable sandwich: RMSNorm → attention → residual → RMSNorm → SwiGLU → residual. Same shape in/out so you can stack \`N\` times.

### 2. Why the LLM pipeline needs it

Parts alone aren’t a deep LM. The block is the unit of depth: residuals keep a path for the original signal; pre-norm keeps scales sane at each entry.

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Wire** | Sandwich above; attention may take a KV cache |
| **Depth** | Llama 3.2 1B: **16** blocks |
| **Train** | Each block has its own weights |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Residual shapes match | Can add |
| **Hard** | Pre-norm placement | Llama contract |
| **Hard** | Per-layer cache if used | No cross-talk / missing layers |
| **Soft** | Still trainable when deep | Residuals + norms |

### 5. What labs ship · what we do

Llama-style pre-norm blocks.  
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

### 1. What it is for

The **stack** wires embed → \`N\` blocks → final RMSNorm → LM head into a full next-token model:

\`ids → embed → Block×N → final RMSNorm → Linear(vocab) → logits\`

Last-dim scores are next-token **logits**. Small Llamas often **tie** LM head ↔ embedding.

### 2. Why the LLM pipeline needs it

One block isn’t a LM. Depth mixes context; the LM head maps hidden states back to **the same vocab** as the tokenizer so you can sample the next id.

### 3. Where it comes from · how you build it

| Part | Role |
| --- | --- |
| Embed | id → vector |
| Blocks × N | mix + transform |
| Final norm | steady before the head |
| LM head | vector → vocab scores (maybe tied) |

Config sets depth/dim/heads; random weights mean a correct graph but nonsense outputs (next chapter loads).

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | logits \`(B, T, vocab)\` | Align with vocab |
| **Hard** | Tying matches claim | Shared params if tied |
| **Hard** | Depth/dim match config | Else official weights won’t fit |
| **Soft** | Coherent text after load | Full-model quality |

### 5. What labs ship · what we do

Llama-style dense end-to-end.  
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

### 1. What it is for

**Loading weights** copies trained tensors into the modules you built. A correct graph with random params is nonsense; official Llama 3.2 weights make the same forward a language model.

### 2. Why the LLM pipeline needs it

“Knowledge” lives in the checkpoint. Infer / align / fine-tune all start from a **correct load**. Wrong keys, tokenizer, or dtype → silent junk or crashes.

### 3. Where it comes from · how you build it

| Step | What happens |
| --- | --- |
| **1** | Build empty modules from config |
| **2** | Download HF Llama 3.2 (license + login) |
| **3** | Copy tensors via a **key map** into arche names |
| **4** | Use the matching official tokenizer |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Param count / shapes match | Missing layers or wrong dims |
| **Hard** | Key map covers every tensor | Unmapped = still random |
| **Hard** | Tokenizer pairing | See tokenizer / embedding |
| **Soft** | Logits close to HF on a short prompt | Alignment chapter |

### 5. What labs ship · what we do

Official HF Llama 3.2 checkpoints.  
\`arche/load_weights.py\` + this episode’s script.

## Run

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

## Fix

Same as tokenizer: license, login, mirror.

## Next

**Text generation** (\`?c=generate\`).

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

### 1. What it is for

**Generation** loops “one forward → next id”: last-position logits → optional temperature / top-k → greedy or sample → append → stop on EOS or length.

### 2. Why the LLM pipeline needs it

One forward only yields the **next-token** distribution, not a full reply. Chat, completion, tools all sit on this loop; products differ in stopping, sampling, and templates.

### 3. Where it comes from · how you build it

| Step | What happens |
| --- | --- |
| **1** | Prefill / forward → logits |
| **2** | Last position; optional temperature, top-k |
| **3** | argmax or sample a new id |
| **4** | Append; stop on EOS or max length |

KV cache (next) makes step 1 cheap after prefill; the **loop** stays the same.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Stops cleanly | EOS / max length work |
| **Hard** | Only past prefix | No future tokens |
| **Soft** | Temperature / top-k behave | 0 greedier; higher more random |
| **Soft** | Coherent text with real weights | Full model + sampling |

### 5. What labs ship · what we do

Every chat stack uses this loop shape.  
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

### 1. What it is for

**KV cache** stores per-layer K/V after prefill. Each decode step only computes the new position’s K/V, appends, and attends over the cache—no full-prompt recompute every step.

### 2. Why the LLM pipeline needs it

Naive decode recomputes K/V over the whole prefix every step. Generation is already a loop; cache makes that loop practical. Without it, long chats are painfully slow and expensive.

### 3. Where it comes from · how you build it

| Stage | What happens |
| --- | --- |
| **Prefill** | One forward; write K/V per layer |
| **Decode** | Append one step; attend to cache |
| **Layout** | This course: concat in time; production may page (later) |
| **Shrink** | GQA / MLA reduce what you store |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Matches full recompute (up to float noise) | Cache must not change meaning |
| **Hard** | Append per layer, no drops | One store per layer |
| **Soft** | Cost grows with **new** tokens, not full prefix each step | The win |
| **Soft** | Memory acceptable | Then GQA / paging |

### 5. What labs ship · what we do

Every production decoder caches KV.  
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

### 1. What it is for

**RoPE scaling** adjusts rotations when context exceeds the training length—Llama 3 style: large \`rope_theta\` plus factor/low/high **wavelength smoothing**.

### 2. Why the LLM pipeline needs it

RoPE angles were tuned for the train length. Naively huge positions distort relative distance. Products wanting 128K-class context often add a scaling schedule (and/or train longer) without inventing a new position system.

### 3. Where it comes from · how you build it

| Path | Plain meaning |
| --- | --- |
| **Llama 3 wavelength smooth** | This lesson; HF-compatible |
| NTK / YaRN / … | Other long-context recipes |
| Train longer | Not formula-only |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Matches checkpoint scaling config | Wrong recipe → long-context junk |
| **Soft** | Near distances still familiar | Don’t break short deps |
| **Soft** | Far distances usable | Long-context evals |

### 5. What labs ship · what we do

Llama 3.2 official long-context schedule.  
Scaling helpers in \`arche/rope.py\` + this script.

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

### 1. What it is for

**MoE** = many expert FFNs (here SwiGLU) + a **router**: each token runs only **top-k** experts—sparse compute for more capacity.

### 2. Why the LLM pipeline needs it

Dense MLPs charge every token full width. MoE aims for huge total params while each token pays for only a few experts (at the cost of harder train/serve).

### 3. Where it comes from · how you build it

| Step | What happens |
| --- | --- |
| **1** | N experts + router |
| **2** | Score tokens; pick top-k |
| **3** | Run chosen experts; weighted merge |
| **Train** | Often load-balancing aux losses |

This lesson is a teaching MoE API—not a full Mixtral stack.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Valid expert indices | No OOB routes |
| **Hard** | top-k matches impl | Shapes/merge correct |
| **Soft** | Load not extremely skewed | Critical in real training |
| **Soft** | Quality vs compute trade | Product evals |

### 5. What labs ship · what we do

Mixtral-style top-k; shared-expert variants exist.  
Primer in \`arche/moe.py\`.

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

### 1. What it is for

**MLA-style** attention caches a **low-rank latent** (often plus a small RoPE stream) instead of full per-head K/V, expanding when needed—less decode memory.

### 2. Why the LLM pipeline needs it

GQA already cuts KV heads; long context can still dominate VRAM. DeepSeek-like paths compress what you cache, with more implementation detail.

### 3. Where it comes from · how you build it

| Idea | Role |
| --- | --- |
| GQA | Fewer KV heads |
| **MLA** | Latent KV |
| Paged KV | How memory is chunked |

Teaching shapes/API here—**not** a bit-exact DeepSeek port.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Latent cache smaller than full KV | Compression real |
| **Hard** | Forward runs; shapes consistent | Primer bar |
| **Soft** | Quality near full attention | Product claim; not this lesson |

### 5. What labs ship · what we do

Primer in \`arche/mla.py\`.

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

### 1. What it is for

Make the model **run and chat**: pick a device, shrink weights, format role messages with a chat template. Not new operators—required glue.

1. **Device** — CUDA → MPS → CPU  
2. **INT8 (teaching)** — smaller weights; dequant then compute  
3. **Chat template** — roles → model string (HF template or text fallback)

Try the page demo.

### 2. Why the LLM pipeline needs it

A correct trunk still fails users without a device, enough RAM, or the right message format. Device = can run; quant = can fit; template = “chat” vs raw completion.

### 3. Where it comes from · how you build it

| Topic | Approach | Production note |
| --- | --- | --- |
| Device | Probe backends; \`.to(device)\` | Same graph, different backends |
| Quant | Teaching INT8; products use INT4/FP8/… | Accuracy vs speed |
| Template | Model’s own chat template | **Don’t invent** formats |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Fallback chain works | No CUDA shouldn’t hard-crash |
| **Hard** | Template matches model | Wrong template → junk |
| **Soft** | Quality after quant OK | Evals; teaching INT8 ≠ product |
| **Soft** | Clear multi-turn roles | System/user/assistant boundaries |

### 5. What labs ship · what we do

Teaching helpers + script—not a full quant product.

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

### 1. What it is for

**Continuous batching (teaching)**: many short decode jobs share one step loop; finished jobs retire so new ones can enter—less “one loop per user, fast waits for slow.”

Try the page demo. **Not** full paged / vLLM.

### 2. Why the LLM pipeline needs it

Servers serve many users. One loop each → GPU idle when finish times differ. Continuous batching turns “who’s still active this step” into scheduling; real systems also need paged KV and fused kernels.

### 3. Where it comes from · how you build it

| Layer | Role |
| --- | --- |
| **Scheduler** | Who’s active each step (this lesson) |
| **Paging / kernels** | How memory/compute run (later primer) |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Finished seqs retire | Don’t keep stepping them |
| **Hard** | No cross-talk | Batched ≠ mixed semantics |
| **Soft** | Fuller GPU, steadier latency | Serving metrics; shape only here |

### 5. What labs ship · what we do

Teaching schedule shape + this script.

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

### 1. What it is for

**Pretraining** teaches next-token prediction on lots of text: logits → cross-entropy vs the next id → update weights (including the embedding table). This is where most trunk parameters are born.

### 2. Why the LLM pipeline needs it

Random weights can’t speak. Learn statistics/structure on huge text first; then chat and preferences. Without pretrain, SFT/DPO have no real starting point.

### 3. Where it comes from · how you build it

| Scale | Meaning |
| --- | --- |
| **This lab** | Toy AdamW: few steps, feel the API |
| **Page playground** | Drag lr/steps; watch loss |
| **Real pretrain** | Massive data, multi-GPU, long runs |

Intuition: batch → predict next token → loss → \`optimizer.step()\`.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Loss is next-id CE | Right objective |
| **Hard** | Grads hit the right params | Including embeddings |
| **Soft** | Loss trends down | Enough on the toy |
| **Soft** | Real runs: data, stability, downstream | Not claimed here |

### 5. What labs ship · what we do

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

### 1. What it is for

Post-pretrain alignment APIs in this lesson:

- **SFT**: prompt→answer demos; **mask the prompt**; loss only on reply tokens  
- **DPO**: vs a frozen reference, prefer chosen over rejected (one teaching step)

### 2. Why the LLM pipeline needs it

Pretrain completes web text; it doesn’t automatically “be an assistant.” SFT teaches format/following; DPO/RLHF-style methods push toward preferences. Chat products need some of this.

### 3. Where it comes from · how you build it

| Method | Data | Goal |
| --- | --- | --- |
| SFT | instruction–reply pairs | Imitate replies |
| DPO | chosen/rejected for a prompt | Preference (no explicit RM in this recipe) |
| Broader map | RLHF / RLAIF / … | Many recipes; two APIs here |

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | SFT mask correct | No loss on the prompt |
| **Hard** | DPO has reference + pairs | API shape |
| **Soft** | Better instruction-following | Human/auto evals |
| **Soft** | Don’t erase pretrain skill | Data/lr tradeoffs |

### 5. What labs ship · what we do

\`arche/train/sft.py\` (and DPO) + this script.

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

### 1. What it is for

**Frontier checkpoint primer**: save/load MoE/MLA-style modules—round-trip a teaching checkpoint and map Mixtral-like \`w1/w2/w3\` names into our modules so shapes don’t silently break.

### 2. Why the LLM pipeline needs it

Running MoE/MLA forward isn’t enough; open weights arrive as foreign key names and shards. Can’t load ⇒ frontier structure stays on paper.

### 3. Where it comes from · how you build it

Mapping + round-trip for open MoE-style checkpoints—not a full trainer.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Round-trip \`allclose\` / shapes | No silent mis-load |
| **Hard** | Name map covers experts | Missed map = random expert |
| **Soft** | Hooks real open weights | Product engineering; shapes here |

### 5. What we do

This episode’s primer script.

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

### 1. What it is for

**HF align**: same short prompt through arche and official HF Llama 3.2; compare argmax and max abs logit diff—evidence the dense trunk and weight map are trustworthy.

### 2. Why the LLM pipeline needs it

Hand-written stacks can “run” with wrong layers. Matching a reference on short logits is hard proof. No weights → clean skip ≠ broken logic.

### 3. Where it comes from · how you build it

This script + CI synthetic map tests; license/login for real weights.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Logits close enough with weights | Map/formulas OK |
| **Hard** | Skip clearly without weights | Don’t fake pass |
| **Soft** | Longer seqs / more ops | Later; short prompt here |

### 5. What we do

See Run; Fix if gated.

## Run

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

## Fix

No weights / not logged in → skip or tip; not a logic failure.

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

### 1. What it is for

Huge checkpoints often ship as \`index.json\` + many shards. This lesson uses **synthetic** index+shards (Mixtral-like MoE / DeepSeek-like MLA) to practice pack/unpack and \`allclose\` after load—no multi-tens-of-GB download.

### 2. Why the LLM pipeline needs it

Single files don’t scale. Can’t read shards ⇒ can’t use frontier open weights. Synthetic data trains **index + concat**, not download stamina.

### 3. Where it comes from · how you build it

Build synthetic shards → load via index → numeric check.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Index matches shards | Missing/wrong keys fail clearly |
| **Hard** | Reload \`allclose\` | Concat correct |
| **Soft** | Real hub formats | Product engineering |

### 5. What we do

This episode’s script.

## Run

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`

## Next

**Paged KV · CUDA Graph** (\`?c=paged-cuda-graph\`).

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

### 1. What it is for

Two serving primers:

1. **Paged KV** — fixed-size blocks + block table + gather  
2. **CUDA Graph** — capture fixed-shape work when CUDA is present  

Teaching mechanisms—**not** a drop-in vLLM.

### 2. Why the LLM pipeline needs it

Toy loops are fine solo; production concurrency fragments VRAM and pays launch tax. Paging manages memory; graphs cut repeated fixed-shape launch overhead.

### 3. Where it comes from · how you build it

\`arche/paged_kv.py\` (and graph) shows block-table gather and optional CUDA Graph capture.

### 4. Standards · what “good” means

| Kind | Check | Plain meaning |
| --- | --- | --- |
| **Hard** | Gather matches logical cache | Paging must not change meaning |
| **Hard** | Block table in bounds | Alloc/free correct |
| **Soft** | Less fragmentation / fewer launches | Serving metrics; shape here |

### 5. What we do

This primer script. Path complete—revisit weak chapters or run HF align with weights.

## Run

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

## Next

Path complete—revisit weak chapters, or run HF align when you have weights.

`,
  },
];
