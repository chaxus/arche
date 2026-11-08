import type { Chapter } from './types';
import { PART0_EN } from './part0';

export const CHAPTERS_EN: Chapter[] = [
  ...PART0_EN,
  {
    id: 'overview',
    ep: '01',
    title: 'Overview',
    summary: 'Why Llama 3.2 ~1B as the example; one from-zero next-token spine.',
    script: 'chapters/01-inference/ep01_overview.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Idea

This course is **one path from zero toward frontier-ready**. Reading order: build the dense trunk → **train weights from zero** → serve faster → frontier structures (MoE/MLA). The running example is **Llama 3.2 ~1B**—wrong example recipe ⇒ later “how industry does it” won’t match.

### 1. What it is for

You hand-write a **Llama 3.2-style** next-token stack—tokenize → embed → blocks×N → LM head (**dense decoder**). Later chapters hang on that same spine. The official Llama files are the example’s finished pair, not a second curriculum.

### 2. Why Llama 3.2 ~1B as the example

**Headline:** learn on a **frontier dense architecture**, at a size you can actually finish.

Llama 3.2 is the small open release of Meta’s **Llama 3** line—the same recipe family used in larger Llama (and echoed across Qwen-class dense stacks): **RMSNorm + RoPE + GQA + SwiGLU + byte-level BPE**. That is today’s mainstream open dense trunk, not a simplified “course-only” net. MoE / MLA and bigger engines come later as primers on top of this trunk.

| Why this pick | Plain meaning |
| --- | --- |
| **Frontier-shaped bricks** | Same module names and contracts you meet in current open dense models and serving stacks |
| **One family, teachable scale** | ~1B keeps V / D / depth / heads countable and HF-downloadable—architecture stays Llama 3, size is for learning |
| **Open finished pair** | Official tokenizer + checkpoint so handwritten vs industry is a real check |
| **Dense first** | Master the dense decoder before MoE/MLA maps; don’t start on a museum GPT-2 |

So: **architecture = current frontier dense; scale = ~1B so from-zero is finishable.** Not “the smartest chatbot,” and not “train 70B at home.”

### 3. Where DeepSeek fits (same path, later map)

When people say **DeepSeek’s frontier open stack** (V2 / V3 / R1-on-V3), they usually mean **not** “another Llama dense.” Headline:

| | **Llama 3.2 (this course’s example)** | **DeepSeek-V2 / V3 line** |
| --- | --- | --- |
| Capacity | **Dense** FFN every layer | **MoE** — many experts, top-k active |
| Attention / KV | **GQA** | **MLA** — latent-compressed KV |
| Shared DNA | RMSNorm / RoPE / modern decoder shape | Same Transformer family, different upgrades |
| In this course | Parts 1–3 hand-write and run | Part 4 primers: MoE → MLA → checkpoint map |

**Read it as:** Llama teaches the dense trunk; DeepSeek is a major open **MoE + MLA** branch on top of that family. We do not start from DeepSeek day one—you need the dense stack first.

### 4. From zero: how you “generate” this path

| Step | What you do |
| --- | --- |
| **1. Fix the example** | Llama 3.2 ~1B |
| **2. Learn how its kind of tokenizer/embeds are made** | Teach-BPE + toy embed; record V, D, depth, heads |
| **3. Hand-write the same bricks** | RMSNorm, RoPE, GQA, SwiGLU… |
| **4. Random shapes first, then train (Part 2) and/or load the Llama example to check** | Graph → numbers |
| **5. Generate + HF check → serve → frontier** | One sidebar path |

### 5. What “good example” means

| Kind | Check |
| --- | --- |
| **Hard** | Config consistent; bricks match the claimed recipe |
| **Hard** | Supports from-zero: tables trainable, net stackable, weights loadable |
| **Soft** | Docs + obtainable weights |
| **Soft** | Room for MoE/MLA primers (DeepSeek-shaped map) |

### 6. Improve · whole-model impact

Swap the example (e.g. Qwen) ⇒ retarget tokenizer/embeds/blocks. Wider/deeper ⇒ full retrain. We keep 1B and use primers for larger ideas.

### 7. Industry · open models

Llama / Qwen ship dense (and some MoE variants); **DeepSeek-V3** is the open flagship of **large MoE + MLA**. All ship config + checkpoint + tokenizer; engines like vLLM assume such checkpoints.

### 8. This course

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

## Next

**Tokenizer** (\`/tokenizer/\`): generate the tokenizer table from zero.

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

This course aims to **build from zero toward frontier-ready**. The tokenizer table is not a disposable download—it is the first table you **choose or train** when you train a model; embeddings, pretraining, and inference cost all bind to it.

### 1. What the tokenizer table is for

Tokenizer table = **cut rules + vocab (piece ↔ id)**.

1. **Cut** — text → tokens  
2. **Number** — tokens → fixed integer ids  

Encode before the model; decode with the **same** table. Specials (BOS/EOS/tools) are ids too.

\`\`\`mermaid
flowchart LR
  text[text] --> cut[cut]
  cut --> ids[token ids]
  ids --> emb[embedding row i]
  emb --> stack[rest of net]
  stack --> ids2[new ids]
  ids2 --> back[decode]
\`\`\`

No table ⇒ embedding has no row indices; next-token pretraining has no vocab.

### 2. From zero: how you generate the table

Open decoders mostly use **byte-level BPE** (GPT/Llama family). From-scratch order:

| Step | What you do | What it fixes |
| --- | --- | --- |
| **1. Corpus mix** | code / EN / CJK / math / web… | which pieces enter the vocab; which languages compress |
| **2. Pretokenizer regex** | spaces, digits, punct, CJK | structure; Chinese UX |
| **3. Base symbols** | UTF-8 **256 bytes** | universal coverage, reversible |
| **4. BPE merges** | merge most frequent adjacent pairs to target size | compression |
| **5. Specials** | BOS/EOS/pad/tool marks | chat/agent protocols |
| **6. Freeze + version** | write \`tokenizer.json\` / merges | embedding rows & checkpoints lock to this id space |
| **7. Then embed + pretrain** | set \`vocab_size\`, build embedding, train | **changing the table ≈ rebuild embeds + heavy retrain** |

This course (same path):

- **How a table is grown:** \`arche/teach_bpe.py\` trains a tiny table so you see generation  
- **The Llama example’s finished table:** official Llama 3.2 tokenizer paired with its weights

### 3. What counts as a good table

**Hard constraints**

| Check | Plain meaning |
| --- | --- |
| **Reversible** | encode→decode recovers normal text |
| **Coverage** | any byte string representable |
| **Stable** | same text → same ids |
| **Train-time identity** | same table for pretrain/finetune/infer |

**Optimization targets**

| Check | Plain meaning | Impact on the whole model |
| --- | --- | --- |
| **Compression / fertility** | fewer tokens better; fertility = pieces/word | longer effective context; cheaper train/infer |
| **Parity** | similar meaning shouldn’t explode across languages | unfair ⇒ short effective context, expensive, undertrained |
| **Domain fit** | code/math/CJK not shredded | vertical ceiling |
| **Structure** | indent, JSON, URLs, numbers | tools & code |
| **Vocab size** | often 32K–200K+ | shorter seqs vs **embed params ∝ V×D** and fatter softmax |
| **Special design** | clear chat/tool boundaries | alignment & agents |
| **Downstream** | loss/evals under same trunk | one final referee |

“Good” is a tradeoff among compression, parity, size, and downstream—not one score.

### 4. How to improve · impact on the whole LLM

| Lever | How | Whole-model impact |
| --- | --- | --- |
| **Corpus mix** | more code/CJK/math | those domains compress & learn better; others may regress |
| **Pretokenizer** | digit/CJK-friendly regex | often beats blindly growing V |
| **Larger V** | more merges | shorter sequences; fatter embed/LM head; more VRAM |
| **Append specials** | grow table tail; keep old ids | **expand embedding rows** and keep training |
| **Change algorithm family** | BPE ↔ Unigram… | **new table**; rebuild embeds; retrain |
| **Extend inherited table** | append domain pieces | cheaper than full reset |

Frontier-from-zero practice: freeze tokenizer after data goals are clear → set \`vocab_size\` → embed → pretrain. Mid-flight table swaps are extremely expensive.

### 5. Industry · well-known open models

Most sit in **byte BPE / tiktoken-like** land; differences are corpus and regex, not “words vs characters” as the main split.

| Model / line | Engine & size | Notes |
| --- | --- | --- |
| **GPT-4 / 3.5** | tiktoken \`cl100k\` ~100K | open cut library; table bound to model |
| **GPT-4o** | \`o200k\` ~200K | generation bump; **new table ≠ old embeds** |
| **Llama 3 / 3.2** | byte BPE 128K+ | **course example**; factory-paired with weights |
| **Qwen** | BPE ~150K-class | self-trained; multilingual + tool specials |
| **DeepSeek-V3** | byte BPE ~128K | pretok/corpus for multilingual compression |
| **Kimi K2** | tiktoken-like ~163K | Han-friendlier regex |
| **Legacy BERT** | WordPiece | encoder era; **not interchangeable** with decoder BPE |
| **Some multilingual** | Unigram + SentencePiece | still used; different rules |

Consensus: trainers ship one table with the recipe; users only load that checkpoint’s tokenizer; next gens often grow V and improve multilingual/code regex.

### 6. Code (this course)

\`\`\`bash
python chapters/01-inference/ep02_teach_bpe.py
\`\`\`

\`\`\`python
from arche.tokenizer import load_tokenizer
tok = load_tokenizer()
ids = tok.encode("The cat sat on the mat.")
\`\`\`

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep02_teach_bpe.py
huggingface-cli login
python chapters/01-inference/ep02_tokenizer.py
\`\`\`

Optional: \`export HF_ENDPOINT=https://hf-mirror.com\`

## Fix

Official table: license / login / mirror. Teaching BPE needs no network.

## Next

**Embedding** (\`/embedding/\`): after the vocab is frozen, how to build the embedding table from zero, judge it, improve it, and what open models do.

`,
  },
  {
    id: 'embedding',
    ep: '03',
    title: 'Embedding',
    summary: 'Id → lookup a float row; why the net needs floats, not dictionary glosses.',
    script: 'chapters/01-inference/ep03_embedding.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    demo: 'embedding',
    markdown: `## Idea

The tokenizer only emits **integer ids**. A neural net cannot “understand” the number \`42\`—it only adds/multiplies tensors. So there must be a step: **use the id to look up a row of floats** in a big table. That table is the **embedding**.

### 1. How does an id become floats—and why bother?

Think **locker wall**, not a dictionary gloss:

1. Tokenizer says: this text became ids \`7, 3, 7\`
2. There are \`V\` lockers (vocab size); each holds exactly **\`D\` floats** (\`D=2048\` in Llama 3.2 1B)
3. **Id → floats = open that locker and copy the row out** (\`gather\` / \`nn.Embedding\`)  
   - id \`7\` → row 7’s \`D\` numbers  
   - id \`3\` → row 3  
   - the same id **always** hits the same row (random at init; training rewrites the numbers)

Tiny fake example (\`D=3\`):

| id | embedding row |
| --- | --- |
| 3 | \`0.2, -0.1, 0.5\` |
| 7 | \`-0.3, 0.8, 0.1\` |

Ids \`[7, 3, 7]\` → three rows: row 7, then row 3, then row 7 again. **No formula turns 7 into floats**—it is a lookup.

**Why floats at all?**

| With only ids | With embeddings |
| --- | --- |
| Ids are labels; you cannot really matmul them with weights | Attention / FFN **only eat float tensors** |
| \`cat=3\` vs \`dog=9\` has no geometry | Training nudges rows so co-occurring tokens get useful starting faces |
| Renumbering the tokenizer breaks everything | Row index must stay locked to tokenizer id |

One line: **tokenizer numbers tokens; embedding gives each number a trainable vector the rest of the net can compute on.** Bad table ⇒ later layers train on noise; wrong pairing with the tokenizer ⇒ the pipeline is junk.

**Where do those floats come from?**  
At first they are **random** (almost no meaning). Then the whole net trains on next-token prediction; wrong guesses backprop and **rewrite rows that showed up**—frequent tokens move most. Results are saved in the checkpoint; inference only loads and gathers. This course can also load Llama’s already-trained rows.

**Why a float row per token at all?**  
Later layers are continuous math (matmul, add, nonlinearities). Integer ids are not valid inputs for that; float rows are. Training tends to push co-occurring / similar-role tokens into vectors that help “guess the next token”—a side effect of the objective, not hand-drawn word maps.

The playground on this page is exactly “gather by id.”

\`\`\`mermaid
flowchart LR
  ids[tokenizer ids] --> gather[lookup rows]
  gather --> floats[T×D floats]
  floats --> blocks[later layers eat floats only]
\`\`\`

### 2. From zero: how you generate the table

**Not** hand-filled glosses, **not** another BPE. BPE assigns ids; embedding stores the float rows—mostly **written by pretrain**.

| Step | What you do | Note |
| --- | --- | --- |
| **1. Lock V** | \`V = tokenizer size\` (incl. specials) | same factory as previous chapter |
| **2. Choose D** | e.g. **2048** for Llama 3.2 1B | architecture hyperparameter |
| **3. Build module** | \`nn.Embedding(V, D)\` | random init |
| **4. Tie or not** | small models often tie embed ↔ lm_head | larger sometimes untie |
| **5. Pretrain** | next-token CE; grads update **seen rows** | grows with the whole net |
| **6. Save** | \`embed_tokens.weight\` in checkpoint | infer = load + gather |
| **7. Changes** | new specials → **grow rows + train**; new tokenizer → **rebuild** | costs differ by orders of magnitude |

Init is an empty locker wall; **training writes the numbers**. This lesson’s toy table shows gather + one update; real tables come from official weights or your pretrain.

### 3. What counts as a good table

**Hard constraints**

| Check | Plain meaning |
| --- | --- |
| **Shape** | \`(V, D)\` matches config/tokenizer; \`T\` ids → \`T\` rows |
| **Pairing** | row \`i\` ↔ tokenizer id \`i\` |
| **Healthy values** | no NaN/Inf; RMSNorm can steady scales |
| **Contract** | this stack: RoPE later; no absolute PE on embeds |

**Optimization targets** (rarely a solo “embedding score”)

| Check | Plain meaning | Whole-model impact |
| --- | --- | --- |
| **Fits pretrain objective** | frequent rows get updates | base LM skill |
| **Dim budget** | too small → weak; too big → hungry | capacity vs stability |
| **Rare tokens** | long-tail ids barely move | weak rare words |
| **Tying** | tie often helps small models | params & output calibration |
| **Downstream / loss** | same-architecture compare | **main judge** |

Unlike fertility, embedding quality is judged mainly via the **full model** under a fixed setup.

### 4. How to improve · impact on the whole LLM

| Lever | How | Whole-model impact |
| --- | --- | --- |
| **Better pretrain** | data, steps, stability | main path: embeds improve with the net |
| **Change D** | width | global capacity; requires retrain |
| **Tie / untie LM head** | share or split | small models often tie |
| **Grow vocab** | append rows (random/mean/subword init) + continue train | new specials/domains; keep old rows |
| **Change tokenizer** | new V | **rebuild embeds + re-pretrain** (or costly migration) |
| **Warm-start rows** | copy from a related model | can speed convergence; wrong pairing destroys the model |
| **Freeze embeds in FT** | small data | less overfit; may cap expressivity |

From-zero order: freeze tokenizer → set \`V,D\` & tying → random embed + pretrain → append rows only when protocols need new specials. **Never** mix Llama embeds with a Qwen tokenizer.

### 5. Industry · well-known open models

| Model / line | How embeddings are done | Notes |
| --- | --- | --- |
| **Llama 3 / 3.2** | lookup; small sizes often **tie** LM head; **no** absolute PE on embeds | **course example**; RoPE for position |
| **Qwen** | lookup + RoPE; V from their tokenizer | strict pairing |
| **DeepSeek** | lookup; frontier focus on MLA/MoE etc. | multilingual tokenizer makes rows “worth more” |
| **GPT line (public)** | larger V → larger embed; generation bumps rebuild embeds | \`o200k\` isn’t a light patch on \`cl100k\` embeds |
| **BERT era** | token + segment + **absolute position** embeds | unlike modern decoder “pure lookup + RoPE” |
| **MoE LLMs** | usually still a shared lookup; sparsity in FFN experts | embed isn’t the MoE headline, but V still costs |

Consensus: embeddings are a **learnable lookup** factory-paired with the tokenizer; improving them ≈ better pretrain/tokenizer; open checkpoints store the matrix in safetensors—you write the same when training from zero.

### 6. Code (this course)

\`\`\`python
self.weight = nn.Embedding(vocab_size, embed_dim)
return self.weight(token_ids)
\`\`\`

\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`

Expect shape, gather, mean |Δ| after one step, finiteness. Real Llama 1B: \`V≈128256\`, \`D=2048\` in the official checkpoint.

## Run

\`\`\`bash
python chapters/01-inference/ep03_embedding.py
\`\`\`

## Next

**RMSNorm** (\`/rmsnorm/\`): how to steady scales after embedding—critical when training deep nets from zero.

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

From-zero deep nets need scale control after embeddings—RMSNorm is the first “steady the ruler.”

### 1. What it is for

**RMSNorm**: divide by RMS, multiply learnable \`γ\`; **no mean subtract**. Llama-style **pre-norm** before attention and MLP.

### 2. From zero: how you build it

| Step | What you do |
| --- | --- |
| **1** | \`RMSNorm(D)\` at each entry that needs steadying |
| **2** | init \`γ\` ≈ 1 |
| **3** | wire pre-norm before Attention/MLP |
| **4** | train with the net |

### 3. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | same shape; \`ε\`; placement matches recipe |
| **Soft** | with \`γ=1\`, RMS≈1; fewer NaNs |
| **Soft** | full-model loss is the judge |

### 4. Improve · whole-model impact

LN→RMSN matches Llama/Qwen; pre- vs post-norm decides depth trainability; \`γ\`/lr couple to convergence.

### 5. Industry · open models

Llama 3/3.2 & Qwen: RMSNorm+pre-norm. GPT-2-era teaching stacks: LayerNorm often post-norm.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## Next

**RoPE** (\`/rope/\`): write position into Q/K from zero.

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

From-zero decoders: embedding says “which token”; **RoPE** (or other PE) writes **position** into attention.

### 1. What it is for

Rotate Q/K by position so relative distance enters the dot product—no absolute PE table on embeds.

### 2. From zero: how you build it

Pick \`rope_theta\`, cache cos/sin, \`rotate_half\` + combine on Q/K; scale later for long context.

### 3. What “good” means

Shape unchanged; relative position affects scores; match checkpoint \`theta\`/scaling.

### 4. Improve · whole-model impact

Theta/scaling ↔ long context; absolute PE→RoPE is modern default; don’t mix PE families with foreign weights.

### 5. Industry · open models

Llama/Qwen/DeepSeek: RoPE. Classic Transformer: sinusoidal. Some long-context lines: ALiBi/YaRN/NTK.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## Next

**GQA** (\`/gqa/\`): choose head counts from zero to save KV cache.

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

From-zero architecture: attention layout sets **decode VRAM** vs quality.

### 1. What it is for

**GQA**: more Q heads than KV heads; groups share KV; still causal.

### 2. From zero: how you build it

Choose divisible \`n_q\`/\`n_kv\`, broadcast KV, train; cache stores \`n_kv\`. Llama 3.2 1B: **32Q/8KV**.

### 3. What “good” means

Valid groups; causal; match weight heads; smaller cache than MHA without quality collapse.

### 4. Improve · whole-model impact

MHA richest/costliest; GQA default open midsize; MQA extreme; MLA further compression. Head-count change ⇒ retrain.

### 5. Industry · open models

Llama 3/Qwen: GQA. DeepSeek-V3: MLA path. Classic GPT-2 teaching: MHA.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## Next

**SwiGLU** (\`/swiglu/\`): wire the FFN from zero.

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

From-zero blocks: attention mixes positions; **SwiGLU FFN** transforms each position—modern dense default.

### 1. What it is for

Gated SiLU FFN with three bias-free linears (gate/up/down).

### 2. From zero: how you build it

Pick \`ffn_dim\`, three mats, wire after second pre-norm; MoE experts are often SwiGLUs.

### 3. What “good” means

In/out \`D\`; no bias (Llama contract); judge gains via full-model metrics.

### 4. Improve · whole-model impact

Wider FFN ⇒ capacity/cost; GELU→SwiGLU needs retrain; MoE multiplies experts.

### 5. Industry · open models

Llama/Qwen dense: SwiGLU. Mixtral-style MoE: SwiGLU experts. GPT-2 teaching: GELU MLP.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## Next

**Transformer block** (\`/block/\`).

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

From zero: parts become one **repeatable layer** so you can stack depth.

### 1. What it is for

Pre-norm sandwich: RMSNorm→attn→res→RMSNorm→SwiGLU→res. Same shape in/out.

### 2. From zero: how you build it

Wire the sandwich; per-block weights; pick depth N (Llama 3.2 1B = **16**); stack and train.

### 3. What “good” means

Residuals add; pre-norm placement; isolated caches; still trainable when deep.

### 4. Improve · whole-model impact

Deeper N ⇒ capability/cost; post→pre-norm enables deep from-zero training; parallel attn∥MLP isn’t Llama’s main line.

### 5. Industry · open models

Llama/Qwen: repeat pre-norm blocks. DeepSeek variants change insides; block boundary remains.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep08_block.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep08_block.py
\`\`\`

## Next

**Stack + LM head** (\`/stack/\`).

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

From zero: one block isn’t an LM—you need **embed + N blocks + final norm + LM head**.

### 1. What it is for

\`ids → embed → Block×N → final RMSNorm → Linear(V) → logits\`. Small Llamas often **tie** head↔embed.

### 2. From zero: how you build it

Build shell from V/D/N/heads; wire stack; choose tying; init then load or pretrain.

### 3. What “good” means

Logits \`(B,T,V)\`; config matches weights; tying consistent; quality after load/train.

### 4. Improve · whole-model impact

Depth/width/tie/V are architecture-level—retrain or careful migration. Optimize modules + training, not “stack alone.”

### 5. Industry · open models

Llama/Qwen dense share this skeleton; MoE swaps FFN for experts inside blocks.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

## Next

**Load weights** (\`/weights/\`).

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

Still one path from zero. This chapter loads the **Llama example’s** open checkpoint so your handwritten stack can speak now. Part 2 is where you feel how those floats get **written by training**—most numbers are trained, not hand-filled. Bad loads waste the handwritten stack.

### 1. What it is for

Copy tensors via a key map. Random weights are a correct graph with nonsense outputs; Llama 3.2 weights make the same forward pass behave like a language model.

### 2. From zero: how you do it

| Step on the same path | What you do |
| --- | --- |
| **Load the example (this chapter)** | license→download→key map→official tokenizer |
| **Train later (Part 2)** | toy/real pretrain writes \`state_dict\`; record tokenizer version |
| **Validate** | shapes/missing keys; **HF align** after generate compares short-prompt logits |

### 3. What “good” means

Shapes/counts; full map; tokenizer pair; optional HF logit align.

### 4. Improve · whole-model impact

Fix maps; dtype; resume carefully; wrong tokenizer destroys meaning.

### 5. Industry · open models

HF + safetensors standard; Llama/Qwen/DeepSeek ship that way; engines consume the same.

### 6. This course

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

## Run

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep10_load_weights.py
\`\`\`

## Fix

License, login, mirror—same as tokenizer.

## Next

**Text generation** (\`/generate/\`).

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

From zero: logits aren’t generation until you **loop sample**. Products differ in stopping/sampling, not in having a loop.

### 1. What it is for

Last-position logits → optional temperature/top-k → greedy/sample → append → stop on EOS/length.

### 2. From zero: how you build it

Forward→sample→stop; add KV cache next; chat templates later.

### 3. What “good” means

Stops; past-only; valid ids; sensible sampling; coherence with real weights.

### 4. Improve · whole-model impact

Sampling knobs don’t change weights; stop rules shape UX; constrained decoding for structure; without KV the loop is correct but too slow.

### 5. Industry · open models

Every chat stack loops; vLLM/TGI speed cache/batching. Llama/Qwen APIs expose sampling.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

## Next


**HF align** (\`/hf-align/\`): verify the handwritten stack against HF.

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

From zero to serving: a correct generate loop isn’t enough—**without KV cache long decode is unusable**.

### 1. What it is for

Prefill stores K/V; decode appends one step and attends to the cache.

### 2. From zero: how you build it

Per-layer buffers; prefill write; decode append; later paging + GQA/MLA to shrink.

### 3. What “good” means

Matches full recompute; no layer mixups; cost grows with new tokens; memory OK.

### 4. Improve · whole-model impact

GQA/MLA/paging/KV-quant cut memory; without cache only toy lengths work.

### 5. Industry · open models

Every production decoder caches KV (vLLM, TGI, llama.cpp…).

### 6. This course

\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep12_kv_cache.py
\`\`\`

## Next

**RoPE scaling** (\`/rope-scaling/\`).

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

After short-context from-zero training, products want 128K-class windows—**RoPE scaling** is a recipe extension, not casually stretching indices.

### 1. What it is for

Llama 3-style wavelength smoothing (large theta + factors) so long positions don’t wrap badly.

### 2. From zero: how you build it

Train at base length → enable checkpoint-matched scaling and/or continue long-data train → eval long context.

### 3. What “good” means

Match checkpoint config; near stable; far usable; long-bench OK.

### 4. Improve · whole-model impact

Wrong scaling breaks long (and maybe short) context; bigger windows raise KV cost; scaling ≠ free long-range knowledge.

### 5. Industry · open models

Llama 3/3.2 official long-context; Qwen/DeepSeek own recipes; YaRN/NTK in open long-context work.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep13_rope_scaling.py
\`\`\`

## Next


**Device · INT8 · chat** (\`/device-quant-chat/\`).

`,
  },
  {
    id: 'moe',
    ep: '14',
    title: 'MoE',
    summary: 'Many experts; each token runs only top-k.',
    script: 'chapters/01-inference/ep14_moe.py',
    partId: 'frontier',
    estimatedMinutes: 15,
    markdown: `## Idea

From zero, huge capacity at controlled compute often means **MoE**—sparse experts. **DeepSeek-V3**’s open story is largely **MoE (capacity) + MLA (KV)**; this chapter is the MoE half. Mixtral is another open MoE; our Llama 3.2 example stayed **dense** on purpose.

### 1. What it is for

Many SwiGLU experts + router; each token runs top-k only. Replaces the dense FFN you wrote earlier—same decoder spine, different capacity trick.

### 2. From zero: how you build it

Pick E/top-k; experts + router; aux load-balance; sparse infer; shard experts in checkpoints. This lesson is an API primer.

### 3. What “good” means

Valid routes; correct merge; balanced load; quality/capacity trade OK.

### 4. Improve · whole-model impact

More experts/k ⇒ capacity/compute; balance loss fights collapse; replacing dense FFN is architecture-level; multi-GPU expert parallel is the engineering cost.

### 5. Industry · open models

| Line | Role |
| --- | --- |
| **DeepSeek-V3** | Large **MoE** + **MLA** (next chapter)—flagship open non-dense stack |
| **Mixtral** | MoE with top-2 SwiGLU experts |
| **Qwen-MoE** | Own E/k recipes |
| **Llama 3.2 ~1B (this course)** | Dense first—then you can read DeepSeek’s map |

### 6. This course

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

## Next

**MLA** (\`/mla/\`).

`,
  },
  {
    id: 'mla',
    ep: '15',
    title: 'MLA',
    summary: 'Cache a small latent instead of full K/V.',
    script: 'chapters/01-inference/ep15_mla.py',
    partId: 'frontier',
    estimatedMinutes: 18,
    markdown: `## Idea

From-zero long-context serving: after GQA, KV can still explode—**MLA** (Multi-head Latent Attention) compresses cache with latents. Together with **MoE**, this is the usual one-line picture of **DeepSeek-V2 / V3**: MoE for capacity, MLA for KV.

### 1. What it is for

Store low-rank latents (+ small RoPE stream), expand when needed. Llama / Qwen mainlines stay on **GQA**; DeepSeek’s open frontier line pushes **MLA**.

### 2. From zero: how you build it

Pick latent dim; train MLA forward; cache latents; this lesson is a shape primer, not bit-exact.

### 3. What “good” means

Smaller cache; consistent shapes; quality near full KV at product bar.

### 4. Improve · whole-model impact

Smaller latent ⇒ save memory / risk quality; composable with paging; architecture-level change.

### 5. Industry · open models

| Line | Attention / KV |
| --- | --- |
| **DeepSeek-V2 / V3** | **MLA** (+ MoE body) |
| **Llama 3 / Qwen dense** | **GQA** mainline |
| Engines (vLLM, …) | Adding MLA support over time |

### 6. This course

\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

## Next


**Frontier checkpoints** (\`/frontier-ckpt/\`).

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

From-zero models must **ship to humans**: device, size, chat format—glue, not new layers.

### 1. What it is for

Device fallback; teaching INT8 shrink; chat templates.

### 2. From zero: how you build it

Probe device → pick quant recipe (INT8 toy; GPTQ/AWQ/FP8 in prod) → **model’s own** template → wire with generate/KV.

### 3. What “good” means

Fallback works; template paired; quant quality OK; clear roles.

### 4. Improve · whole-model impact

Harder quant ⇒ smaller/faster/riskier; wrong template ⇒ junk; multi-backend is engineering cost.

### 5. Industry · open models

HF chat templates; vLLM/TGI load quant+template; Llama/Qwen ship official templates; llama.cpp/GGML quants.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep16_device_quant_chat.py
\`\`\`

## Next

**Continuous batching** (\`/continuous-batching/\`).

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

From zero single-user loops → industry **multi-user shared stepping**: continuous batching is the serving schedule shape.

### 1. What it is for

Share a step among jobs; retire finished; admit new. Teaching ≠ full vLLM.

### 2. From zero: how you build it

Active set → step only actives → retire; add paged KV later for production shape.

### 3. What “good” means

Retire finished; no cross-talk; fuller GPU / steadier tail latency.

### 4. Improve · whole-model impact

Better schedule ⇒ more throughput on same hardware; coupled to KV layout, not weights.

### 5. Industry · open models

vLLM continuous batch + paged attention; TGI/TRT-LLM/SGLang same idea.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep17_continuous_batching.py
\`\`\`

## Next


**Paged KV · CUDA Graph** (\`/paged-cuda-graph/\`).

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

From-zero core: **pretrain** writes embeddings and almost all trunk weights. Without it you only have a random shell.

### 1. What it is for

Next-token CE on huge text; backprop updates the net including embeds.

### 2. From zero: how you build it

Frozen tokenizer + shell + data → AdamW next-id loss → long multi-GPU run → checkpoints. This lesson: toy steps + page playground.

### 3. What “good” means

Correct objective; grads hit params; loss/downstream/stability.

### 4. Improve · whole-model impact

Data mix dominates; scale laws; stability decides finishability; undertrain ⇒ weak embeds and deep layers.

### 5. Industry · open models

Llama/Qwen/DeepSeek: pretrain then align; Megatron/DeepSpeed/torchtitan-style stacks for from-zero training.

### 6. This course

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## Run

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## Next

**SFT · DPO** (\`/sft-dpo/\`).

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

End of Part 2: after pretrain you still **align**—SFT teaches following; DPO/RLHF-style methods push preferences. Part 3 then makes decode servable.

### 1. What it is for

SFT mask-prompt loss on replies; DPO prefer chosen vs rejected against a reference.

### 2. From zero: how you build it

Instruction/preference data → SFT from base ckpt → DPO/RLHF → eval regressions.

### 3. What “good” means

Correct masks/shapes; better following/safety; limited capability loss.

### 4. Improve · whole-model impact

Data quality beats blind steps; over-align can dull the base; often paired with new specials/templates.

### 5. Industry · open models

Llama/Qwen Instruct/Chat; Zephyr-style DPO; TRL-like tooling; labs use RLHF variants.

### 6. This course

\`\`\`bash
python chapters/02-training/ep19_sft_dpo.py
\`\`\`

## Run

\`\`\`bash
python chapters/02-training/ep19_sft_dpo.py
\`\`\`

## Next


**KV cache** (\`/kv-cache/\`): make decode affordable before serving.

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

From-zero MoE/MLA must **speak checkpoint names/shapes** of open weights—or frontier structure stays on paper.

### 1. What it is for

Round-trip teaching ckpts; map Mixtral-like \`w1/w2/w3\` into our modules.

### 2. From zero: how you build it

Name map ↔ open names; save/load; fail loudly on missing keys.

### 3. What “good” means

Round-trip align; experts covered; hooks real hubs.

### 4. Improve · whole-model impact

Bad maps ⇒ silent random experts; consistent naming cuts debt.

### 5. Industry · open models

HF shards + documented keys; Mixtral/DeepSeek naming; converters are standard.

### 6. This course

\`\`\`bash
python chapters/02-training/ep20_frontier_ckpt.py
\`\`\`

## Run

\`\`\`bash
python chapters/02-training/ep20_frontier_ckpt.py
\`\`\`

## Next


**Sharded MoE / MLA** (\`/sharded-frontier/\`).

`,
  },
  {
    id: 'hf-align',
    ep: '21',
    title: 'HF logits alignment',
    summary: 'Check our logits against Hugging Face on a short prompt.',
    script: 'chapters/01-inference/ep21_hf_align.py',
    partId: 'trunk',
    estimatedMinutes: 20,
    markdown: `## Idea

End of Part 1: **match authoritative HF logits** on a short prompt—proof the map and formulas are sound. Then Part 2 shows how weights get written from zero.

### 1. What it is for

Same prompt through arche vs HF Llama 3.2; compare argmax / max abs diff.

### 2. From zero: how you build it

Fixed seed/dtype; same weights; short seq; clean skip without weights.

### 3. What “good” means

Close enough with weights; explicit skip without; longer coverage later.

### 4. Improve · whole-model impact

Fail ⇒ revisit maps/RoPE/norm. Pass before optimize/serve.

### 5. Industry · open models

Repro CI often gates on HF/official match; engines keep numeric regressions.

### 6. This course

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

## Run

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

## Fix

No weights/login → skip ≠ broken logic.

## Next


**Pretrain** (\`/pretrain/\`): write weights from zero (toy API).

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

From-zero at scale: weights are **index.json + shards**. Can’t assemble shards ⇒ can’t take open frontier weights.

### 1. What it is for

Synthetic pack/unpack + \`allclose\` without multi-tens-of-GB downloads.

### 2. From zero: how you build it

HF-style index; shard tensors; aggregate on load.

### 3. What “good” means

Index matches shards; reload aligns; missing pieces fail clearly.

### 4. Improve · whole-model impact

Shard policy affects download parallelism and peak RAM; bad concat ⇒ silent bad weights.

### 5. Industry · open models

HF Hub sharding standard; Llama/Qwen/DeepSeek/Mixtral; safetensors+index dominant.

### 6. This course

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`

## Run

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`

## Next


Path complete—revisit weak chapters, or re-run HF align with weights.

`,
  },
  {
    id: 'paged-cuda-graph',
    ep: '23',
    title: 'Paged KV · CUDA graphs',
    summary: 'Block-table KV pool and fixed-shape CUDA graphs.',
    script: 'chapters/01-inference/ep23_paged_cuda_graph.py',
    partId: 'infer',
    estimatedMinutes: 18,
    markdown: `## Idea

End of Part 3: **paged KV** fights fragmentation; **CUDA Graph** cuts fixed-shape launch tax—toward vLLM-class engines. Next part enters MoE/MLA frontier structures.

### 1. What it is for

Block table + gather; optional CUDA Graph capture. Teaching ≠ drop-in vLLM.

### 2. From zero: how you build it

Logical KV + continuous batch → block alloc/free → optional graph on decode.

### 3. What “good” means

Gather matches logical cache; in-bounds tables; less fragmentation/launch; higher throughput.

### 4. Improve · whole-model impact

Page size/prealloc change waste vs speed; graphs need stable shapes; doesn’t change weights—raises serving ceiling.

### 5. Industry · open models

vLLM paged attention landmark; TRT-LLM/SGLang variants; pairs with Llama/Qwen/DeepSeek weights.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

## Next


**MoE** (\`/moe/\`): frontier architectures.

`,
  },
];
