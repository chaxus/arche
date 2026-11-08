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

This course trains **from zero toward frontier-ready**: pick a production-isomorphic specimen, hand-write blocks, then speed and train. Wrong specimen ⇒ later “how industry does it” won’t match.

### 1. What it is for

Specimen: **Llama 3.2-style** next-token stack—tokenize → embed → blocks×N → LM head (**dense decoder**). Later chapters hang here.

### 2. From zero: how you “generate” this path

| Step | What you do |
| --- | --- |
| **1. Pick open, downloadable, small, production-like** | Here: Llama 3.2 ~1B |
| **2. Freeze tokenizer; record V, D, depth, heads** | Config is the contract |
| **3. Hand-write the same bricks** | RMSNorm, RoPE, GQA, SwiGLU… |
| **4. Random shapes first, then load/train weights** | Graph → numbers |
| **5. Add serving & alignment** | KV, batching, SFT… |

### 3. What “good choice” means

| Kind | Check |
| --- | --- |
| **Hard** | Config consistent; bricks match the claimed recipe |
| **Hard** | Supports from-zero: tables trainable, net stackable, weights loadable |
| **Soft** | Docs + obtainable weights |
| **Soft** | Room for MoE/MLA primers |

### 4. Improve · whole-model impact

Swap specimen (e.g. Qwen) ⇒ retarget tokenizer/embeds/blocks. Wider/deeper ⇒ full retrain. We keep 1B and use primers for larger ideas.

### 5. Industry · open models

Llama / Qwen / DeepSeek ship config + checkpoint + tokenizer; engines like vLLM assume such checkpoints.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

## Next

**Tokenizer** (\`?c=tokenizer\`): generate the tokenizer table from zero.

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

This course:

- **Teaching:** \`arche/teach_bpe.py\` trains a tiny table so you see generation  
- **Specimen:** official Llama 3.2 table for a production pairing

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
| **Llama 3 / 3.2** | byte BPE 128K+ | **course specimen**; factory-paired with weights |
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

**Embedding** (\`?c=embedding\`): after the vocab is frozen, how to build the embedding table from zero, judge it, improve it, and what open models do.

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

After the tokenizer is frozen, the next from-scratch step is the **embedding table**: each id becomes a trainable float row. A bad table makes every later layer learn on noise; a mismatched tokenizer breaks the whole frontier pipeline.

### 1. What the embedding table is for

Matrix \`(V, D)\`: **row \`i\` = floats for id \`i\`** (\`V\` = vocab size, \`D\` = model dim).

| Role | Plain meaning |
| --- | --- |
| **① Interface** | discrete ids → tensors you can multiply/add |
| **② Learnable identity** | each token’s starting face; pretrain is the main author |
| **③ Often tied LM head** | small models share gather-in / score-out |

Inference ≈ **gather** by id (try the page table). Not a dictionary gloss.

\`\`\`mermaid
flowchart LR
  tok[tokenizer frozen] --> shape["set V = vocab_size"]
  shape --> init["random init V×D"]
  init --> pretrain[pretrain updates rows]
  pretrain --> ckpt[checkpoint]
\`\`\`

### 2. From zero: how you generate the table

**Not** hand-filled, **not** another BPE. BPE assigns ids; embedding stores learnable floats.

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
| **Llama 3 / 3.2** | lookup; small sizes often **tie** LM head; **no** absolute PE on embeds | **course specimen**; RoPE for position |
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

**RMSNorm** (\`?c=rmsnorm\`): how to steady scales after embedding—critical when training deep nets from zero.

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

**RoPE** (\`?c=rope\`): write position into Q/K from zero.

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

**GQA** (\`?c=gqa\`): choose head counts from zero to save KV cache.

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

**SwiGLU** (\`?c=swiglu\`): wire the FFN from zero.

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

**Transformer block** (\`?c=block\`).

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

From zero, two paths meet: **(A) write your own checkpoint via pretrain**; **(B) load an open checkpoint**. Bad loads waste the handwritten stack.

### 1. What it is for

Copy tensors via a key map. Random weights are a correct graph with nonsense outputs.

### 2. From zero: how you do it

Self-train: save state_dict + tokenizer version. Open load: license→download→key map→official tokenizer→validate.

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

From zero, huge capacity at controlled compute often means **MoE**—sparse experts (Mixtral, DeepSeek-MoE, …).

### 1. What it is for

Many SwiGLU experts + router; each token runs top-k only.

### 2. From zero: how you build it

Pick E/top-k; experts + router; aux load-balance; sparse infer; shard experts in checkpoints. This lesson is an API primer.

### 3. What “good” means

Valid routes; correct merge; balanced load; quality/capacity trade OK.

### 4. Improve · whole-model impact

More experts/k ⇒ capacity/compute; balance loss fights collapse; replacing dense FFN is architecture-level; multi-GPU expert parallel is the engineering cost.

### 5. Industry · open models

Mixtral top-2; DeepSeek-V3 large MoE; Qwen-MoE variants; Llama 3.2 1B specimen stays dense first.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

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

From-zero long-context serving: after GQA, KV can still explode—**MLA** compresses cache with latents (DeepSeek line).

### 1. What it is for

Store low-rank latents (+ small RoPE stream), expand when needed.

### 2. From zero: how you build it

Pick latent dim; train MLA forward; cache latents; this lesson is a shape primer, not bit-exact.

### 3. What “good” means

Smaller cache; consistent shapes; quality near full KV at product bar.

### 4. Improve · whole-model impact

Smaller latent ⇒ save memory / risk quality; composable with paging; architecture-level change.

### 5. Industry · open models

DeepSeek-V2/V3 push MLA; Llama/Qwen stay GQA-mainline; engines add support over time.

### 6. This course

\`\`\`bash
python chapters/01-inference/ep15_mla.py
\`\`\`

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
