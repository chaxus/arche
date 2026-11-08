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

This course is **one path from zero toward frontier-ready**. Reading order: tables → **next-token train** → dense bricks → align → serve → frontier (MoE/MLA). The running example is **Llama 3.2 ~1B**—wrong example recipe ⇒ later “how industry does it” won’t match.

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
| **2. Tokenizer + embedding** | Teach-BPE + lookup table; record V, D, depth, heads |
| **3. Next-token train** | Shift + CE + backprop—**writes** embeds and weights |
| **4. Hand-write the bricks** | RMSNorm, RoPE, GQA, SwiGLU… then stack |
| **5. Load Llama / generate / HF → align → serve → frontier** | One sidebar path |

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

**Pretrain** (\`/pretrain/\`): the next-token loop that **writes** those float rows (and the rest of the trunk).

`,
    latex: 'X = \\mathrm{Embed}(t_{1:L}) \\in \\mathbb{R}^{L \\times d}',
  },
{
    id: 'pretrain',
    ep: '18',
    title: 'Pretrain',
    summary: 'Next-token training: shift, CE, backprop—how embeds and weights get written.',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'trunk',
    estimatedMinutes: 15,
    demo: 'pretrain',
    markdown: `## Idea

**What:** *Causal language modeling* / **next-token pretraining** — given tokens to the left, score every vocab id as “what comes next,” then train so the true next id gets high probability.

**Why it matters:** This single self-supervised objective is what **writes** embedding rows and almost all trunk weights. Without it you have a random shell (or you load someone else’s floats, like the Llama example later).

**How (one line):** tokenize text → embed → forward → **shift** so position \`t\` predicts id \`t+1\` → cross-entropy → backward → optimizer step. Repeat on huge corpora.

Industry names the same thing: nanoGPT’s \`X,Y\` windows, Hugging Face “causal LM”, Llama 3 / DeepSeek “pretrain then align.”

### 1. What — the objective

For a token sequence \\(x_1,\\ldots,x_T\\):

\\[
p(x_1,\\ldots,x_T)=\\prod_t p(x_t \\mid x_{<t})
\\]

Training minimizes average **negative log-likelihood** = token-level **cross-entropy**. Labels come from the text itself (self-supervised)—no human “answer” column.

| Name | Meaning |
| --- | --- |
| **logits** | Raw scores \`(batch, time, V)\` before softmax |
| **next-token CE** | At each time, punish low probability on the true next id |
| **causal** | Position \`t\` must not see tokens after \`t\` (mask / decoder stack) |

### 2. Why — why this design

| Choice | Why |
| --- | --- |
| Predict **next** id | Turns any raw text into infinite training signal |
| One forward → many positions | Length-\`T\` sequence yields ~\`T-1\` supervised guesses (causal mask makes each legal) |
| CE on vocab | Matches “pick a discrete token”; differentiable via softmax |
| Update **embeds + all bricks** | Embedding chapter’s floats are not hand-filled—they move here |
| Pretrain **before** SFT/DPO | Align steers an already next-token-capable net; it does not invent the base skill |

Inference vs training:

| | Inference | Training (this chapter) |
| --- | --- | --- |
| True next id | Unknown | Known from the batch |
| After logits | sample / argmax | CE vs true id |
| Weights | usually frozen | updated every step |

### 3. How — shift, loss, one step (real code)

Tiny example ids \`[10, 20, 30]\` (“The cat sat” style, fake ids):

| Position | Sees (conceptually) | Must predict |
| --- | --- | --- |
| 0 | \`10\` | \`20\` |
| 1 | \`10,20\` | \`30\` |

**Implementation in this repo** ([\`arche/train/pretrain.py\`](https://github.com/chaxus/arche/blob/main/arche/train/pretrain.py)):

\`\`\`python
def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Predict token t+1 from positions 0..T-1."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),  # drop last time
        targets[:, 1:].reshape(-1),                   # drop first id
    )
\`\`\`

Same batch tensor is both input and label source: logits at \`t\` align with \`targets[t+1]\`.

Full micro-step:

\`\`\`python
opt.zero_grad(set_to_none=True)
logits, _ = model(batch)       # (B, T, V)
loss = lm_loss(logits, batch)
loss.backward()
opt.step()
\`\`\`

Lab driver ([\`chapters/02-training/ep18_pretrain.py\`](https://github.com/chaxus/arche/blob/main/chapters/02-training/ep18_pretrain.py)): tiny \`Llama\` + random id batches + \`train_pretrain_steps\` → print loss start→end.

**HF pitfall:** \`DataCollatorForLanguageModeling(mlm=False)\` often sets \`labels = input_ids\` **unshifted**; Transformers CausalLM shifts **inside** \`forward\`. Hand-written loops (ours / nanoGPT) shift explicitly. **Do not shift twice.**

\`\`\`mermaid
flowchart LR
  batch[ids_BT] --> fwd[model_forward]
  fwd --> logits[logits_BTV]
  logits --> shift[logits_tm1_vs_ids_t]
  shift --> ce[cross_entropy]
  ce --> bwd[backward]
  bwd --> adam[AdamW_step]
\`\`\`

Sanity: at random init, loss ≈ \`log(V)\` (uniform over vocab).

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Shift is one token; causal (no future leak) |
| **Hard** | Grads reach embed rows + trunk; no NaN |
| **Soft** | Loss falls; samples less nonsense |
| **Soft** | Matches published recipe at scale (later chapters) |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| More / better tokens | Main capability ceiling |
| Correct shift / mask | Silent wrong objective if broken |
| LR / clip / accum | Finishability (next chapter) |
| Under-train | Weak embeds **and** deep layers |

Next lessons split **data windows**, **optimizers**, and **frontier scale recipes** so this chapter stays the objective + one step.

### 6. Industry pointers

- Raschka FAQ: constructing input/target windows  
- nanoGPT \`train.py\`: readable full loop  
- HF LLM Course ch7.6: causal LM from scratch  
- Llama 3 / DeepSeek reports: same family of objective, different scale & architecture  

### 7. This course

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

Page playground: drag lr / steps. Forward stack details come in RMSNorm→… after the data/optim primers.

## Run

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## Next

**Pretrain data** (\`/pretrain-data/\`): how batches and sliding windows are built.

`,
  },
{
    id: 'pretrain-data',
    ep: '18b',
    title: 'Pretrain data',
    summary: 'Windows, packing, batches—how text becomes (B, T) for the CE loop.',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Idea

**What:** How pretraining turns raw text into **\`(batch, time)\` id tensors** the loss can eat—document → tokenize → pack/cut windows → batch → (optional) pad/mask.

**Why:** The objective in *Pretrain* is fixed; **data shape** decides what signal each step sees. Wrong windows (empty pads as labels, leaked future, broken packing) silently waste or corrupt training.

**How:** Build fixed-length contexts (or pack many docs into one stream), form batches, keep **labels aligned with the same shift rule** as your loss.

### 1. What — pieces of a batch

| Piece | Role |
| --- | --- |
| **document** | Raw text (web, books, code, …) |
| **token ids** | Tokenizer output (BPE etc.) |
| **window / block** | Length-\`T\` slice the model sees at once (\`T\` ≈ context) |
| **batch** | Stack \`B\` windows → shape \`(B, T)\` |
| **labels** | Same ids, interpreted with shift (or HF’s internal shift) |

Two common layouts:

| Layout | Idea |
| --- | --- |
| **Sliding / chunked** | Cut long token streams into non-overlap or stride-\`T\` blocks (nanoGPT style) |
| **Packed** | Concatenate many short docs with separators / attention boundaries so GPUs stay full |

### 2. Why — design pressure

| Pressure | Consequence |
| --- | --- |
| GPU likes dense \`(B,T)\` | Prefer packing over tiny ragged sequences |
| Context length \`T\` | Caps how far “next token” can look back |
| Pad tokens | Must **not** train as if they were real next words (mask or ignore_index) |
| Doc boundaries | Blind concat can invent fake “next tokens” across unrelated docs unless you mask |

### 3. How — toy window (matches this repo)

Lab batches are random ids for teaching the **loop**, not real corpora:

\`\`\`python
# ep18-style: B windows of length T, vocab V
batch = torch.randint(0, V, (B, T))
logits, _ = model(batch)
loss = lm_loss(logits, batch)  # uses batch[:, 1:] as targets
\`\`\`

Production sketch (conceptual, nanoGPT-like):

\`\`\`python
# tokens: 1D long stream of ids
ix = torch.randint(0, len(tokens) - T, (B,))
x = torch.stack([tokens[i : i + T] for i in ix])
y = torch.stack([tokens[i + 1 : i + T + 1] for i in ix])
# then CE(model(x), y)  — y already shifted; don't shift again
\`\`\`

**HF:** if collator leaves \`labels == input_ids\`, the **model** shifts; if you build \`y\` yourself, your CE must **not** shift again.

\`\`\`mermaid
flowchart TD
  docs[raw_docs] --> tok[tokenizer]
  tok --> stream[id_stream]
  stream --> win[cut_or_pack_windows]
  win --> bat[stack_batch_BT]
  bat --> loss[same_shift_as_lm_loss]
\`\`\`

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Every supervised position has a real next id (or masked) |
| **Hard** | Label/input shift matches the loss implementation |
| **Soft** | High token utilization (little pure padding) |
| **Soft** | Mix covers the domains you care about |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Cleaner / more tokens | Stronger base model |
| Longer \`T\` | Better long-range next-token skill (cost ↑) |
| Packing quality | Throughput and effective updates/day |
| Domain mix | What the model becomes “fluent” at |

### 6. This course

No separate script yet—reuse:

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

Treat random batches as “fake windows.” Real dataloaders appear when you leave the toy spine.

## Run

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## Next

**Pretrain optim** (\`/pretrain-optim/\`): AdamW, lr, grad clip, accumulation.

`,
  },
{
    id: 'pretrain-optim',
    ep: '18c',
    title: 'Pretrain optim',
    summary: 'AdamW, lr, clip, accumulation—making the CE step stable.',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Idea

**What:** The **optimizer side** of one pretrain step—AdamW, learning rate (and schedules), gradient clipping, and gradient accumulation—so the CE objective actually moves weights stably.

**Why:** Correct loss + bad optim → NaNs, divergence, or “loss stuck.” At any scale past toys, **how you step** decides whether a run finishes.

**How:** \`zero_grad\` → forward → loss → \`backward\` → (optional clip) → \`optimizer.step\` → (optional lr schedule). Accumulate grads over micro-batches when \`B\` must stay small.

### 1. What — knobs

| Knob | Role |
| --- | --- |
| **AdamW** | Adaptive moments + **decoupled weight decay** (default in Llama/nanoGPT-style recipes) |
| **lr** | Step size; too high → explode; too low → crawl |
| **schedule** | Warmup then decay (cosine / WSD / …) so early steps don’t thrash random embeds |
| **grad clip** | Cap global grad norm (e.g. 1.0) against spikes |
| **grad accum** | Sum grads over \`K\` micro-batches ≈ larger effective batch |

This repo’s toy loop ([\`train_pretrain_steps\`](https://github.com/chaxus/arche/blob/main/arche/train/pretrain.py)):

\`\`\`python
opt = torch.optim.AdamW(model.parameters(), lr=lr)
for batch in batches:
    opt.zero_grad(set_to_none=True)
    logits, _ = model(batch)
    loss = lm_loss(logits, batch)
    loss.backward()
    opt.step()
\`\`\`

No schedule/clip yet—on purpose. Production adds them around the same skeleton.

### 2. Why — each piece

| Choice | Why |
| --- | --- |
| AdamW over plain SGD | Sparse-ish token grads + many params; adaptive scales help |
| Weight decay | Mild regularization; AdamW’s form matches LLM practice |
| Warmup | Early CE is huge (\`~log V\`); blunt lr shreds init |
| Clip | One bad batch shouldn’t blow FP math |
| Accum | Fit long \`T\` / big models on limited VRAM while keeping effective batch |

### 3. How — accumulation sketch

\`\`\`python
opt.zero_grad(set_to_none=True)
for micro in micro_batches:
    loss = lm_loss(*model_and_shift(micro)) / K
    loss.backward()          # grads accumulate
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
# scheduler.step()  # if any
\`\`\`

\`\`\`mermaid
flowchart LR
  mb[micro_batches] --> bwd[backward_x_K]
  bwd --> clip[clip_grad_norm]
  clip --> step[AdamW_step]
  step --> sch[lr_schedule]
\`\`\`

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | No NaN/Inf in loss or grads after clip |
| **Hard** | \`zero_grad\` each **optimizer** step (not forgotten under accum) |
| **Soft** | Loss trends down over a sensible window |
| **Soft** | lr schedule matches a known recipe when you scale up |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| lr / warmup | Trainability of deep stacks + embeds |
| effective batch (B×accum×world) | Noise vs stability of updates |
| clip threshold | Spike survival vs under-updating |
| Wrong decay | Over-shrink or under-regularize weights |

### 6. This course

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

Page demo: try higher lr until loss misbehaves—that’s the optim lesson in miniature.

## Run

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## Next

**Pretrain scale** (\`/pretrain-scale/\`): how frontier recipes stretch the same loop.

`,
  },
{
    id: 'pretrain-scale',
    ep: '18d',
    title: 'Pretrain scale',
    summary: 'Same next-token loop at frontier scale: tokens, stages, parallelism.',
    script: 'chapters/02-training/ep18_pretrain.py',
    partId: 'trunk',
    estimatedMinutes: 10,
    markdown: `## Idea

**What:** A **primer** on scaling the same next-token loop—more tokens, stages, parallelism—without leaving the beginner spine for a full systems course.

**Why:** Llama 3.2 ~1B and frontier runs share **objective + step**; they differ in **data volume, staged recipes, cluster math, and (sometimes) architecture** (dense vs MoE). Knowing the map prevents “my toy loss went down, so I’m done.”

**How:** Read published recipes as *extensions* of chapters you already have: same CE, harder data/optim/engineering.

### 1. What — scale axes

| Axis | Toy (this course) | Frontier-shaped |
| --- | --- | --- |
| **Tokens** | A few random batches | Trillions of real tokens |
| **Steps / stages** | One flat loop | Multi-stage LR, data mixes, context lengthening |
| **Batch** | Tiny \`B,T\` | Huge effective batch via accum + data parallel |
| **Model** | Tiny dense \`Llama\` | Dense 1B–70B+ or MoE (DeepSeek) with MLA/experts |
| **Hardware** | One process | Megatron/FSDP/pipeline-style sharding |

Llama 3-style public reports: long pretrain on curated mixes, then shorter high-quality anneal / long-context phases—still next-token at heart.

DeepSeek-V2/V3: **same causal LM family**, different **block** (MLA + MoE)—see Frontier chapters; training still CE + big-data optim.

### 2. Why — why not stop at ep18

| Illusion | Reality |
| --- | --- |
| “Loss fell on random ids” | Proves the **loop**, not language skill |
| “Copy AdamW defaults” | Defaults need warmup/schedule/clip at scale |
| “Bigger model only” | Data quality × tokens often dominate |
| “SFT will fix a bad base” | Align edits style; weak pretrain stays weak |

### 3. How — read a recipe in four lines

When you open Llama / DeepSeek / Megatron writeups, map:

1. **Objective** → still next-token CE (Pretrain)  
2. **Data** → mixes, packing, context schedule (Pretrain data)  
3. **Optim** → AdamW, lr shape, clip, effective batch (Pretrain optim)  
4. **Systems** → parallel strategy, precision, checkpointing (later / ops)

\`\`\`mermaid
flowchart TD
  obj[same_CE_objective] --> data[more_better_tokens]
  data --> opt[stable_large_batch_optim]
  opt --> sys[multi_GPU_sharding]
  sys --> align[then_SFT_DPO]
\`\`\`

### 4. What “good” means (beginner bar)

| Kind | Check |
| --- | --- |
| **Hard** | You can point to which axis a paper changed |
| **Soft** | You don’t confuse MoE **architecture** with a different **loss** |
| **Soft** | You expect engineering chapters before “reproduce Llama” |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Token budget | Primary capability driver |
| Staged data / LR | Stability + late quality |
| Parallel efficiency | Whether the budget is usable |
| Arch (GQA/MoE/MLA) | Cost per token & memory—not a new objective |

### 6. This course

Still the toy entry:

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

Next in the spine: dense **bricks** (RMSNorm→…) so the forward behind \`model(batch)\` is no longer a black box—then HF weight align, then **SFT/DPO**.

## Run

\`\`\`bash
python chapters/02-training/ep18_pretrain.py
\`\`\`

## Next

**RMSNorm** (\`/rmsnorm/\`): first trunk brick inside the forward you just trained.

`,
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

**What:** **RMSNorm** — per-token scale control: divide by root-mean-square, multiply learnable \`γ\`. No mean subtract, no bias.

**Why:** Deep next-token stacks blow up or die without steady scales. Llama-style **pre-norm** (norm → attn / MLP) is what makes depth trainable from zero.

**How:** \`RMSNorm(D)\` at each pre-norm site; \`γ ≈ 1\` at init; train with the net.

### 1. What — formula

\\[
\\mathrm{RMSNorm}(x)=\\frac{x}{\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}}\\odot\\gamma
\\]

Repo ([\`arche/norms.py\`](https://github.com/chaxus/arche/blob/main/arche/norms.py)):

\`\`\`python
variance = x_f.pow(2).mean(dim=-1, keepdim=True)
x_normed = x_f * torch.rsqrt(variance + self.eps)
return (self.weight.float() * x_normed).to(orig_dtype)
\`\`\`

Math in float32, cast back—same stability trick HF Llama uses.

### 2. Why — vs LayerNorm / placement

| Choice | Why |
| --- | --- |
| No mean center | Slightly cheaper; Llama/Qwen recipe |
| Learnable \`γ\` only | Scale per channel after unit-RMS |
| **Pre-norm** | Stabilize *inputs* to attn/MLP so depth survives |
| After embed, before bricks | First “steady the ruler” once vectors exist |

GPT-2-era teaching often: LayerNorm + post-norm. This course’s example is **RMSNorm + pre-norm**.

### 3. How — wire it

| Step | Do |
| --- | --- |
| 1 | \`RMSNorm(D)\` before attention and before SwiGLU (see block) |
| 2 | Final RMSNorm before LM head (see stack) |
| 3 | Match \`ε\` to the checkpoint (\`norm_eps\`) |
| 4 | Train / load; don’t swap LN↔RMSN under foreign weights |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Same shape as input; \`ε\` present; pre-norm sites match recipe |
| **Soft** | With \`γ=1\`, RMS≈1; fewer NaNs in deep forward |
| **Soft** | Full-model loss / HF align is the real judge |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| LN → RMSN | Architecture change ⇒ retrain |
| Pre vs post | Depth trainability |
| \`γ\` / lr | Couples to convergence |

### 6. This course

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep04_rmsnorm.py
\`\`\`

## Next

**RoPE** (\`/rope/\`): write position into Q/K.

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

**What:** **RoPE** (Rotary Position Embedding) — rotate Q and K by token position so **relative distance** enters the attention dot product.

**Why:** Embeddings say *which* token; attention still needs *where*. Absolute PE tables on embeds are older; modern dense Llama/Qwen/DeepSeek use RoPE on Q/K.

**How:** Pick \`rope_theta\`, cache cos/sin, \`rotate_half\` + combine on Q/K; long-context scaling is a later knob.

### 1. What — apply step

Repo ([\`arche/rope.py\`](https://github.com/chaxus/arche/blob/main/arche/rope.py)):

\`\`\`python
def apply_rotary_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
\`\`\`

\`build_rope_cache\` precomputes cos/sin for \`0 .. seq_len-1\`. Attention slices by \`start_pos\` when decoding with cache.

### 2. Why — design

| Choice | Why |
| --- | --- |
| Rotate Q/K only | V stays content; positions live in scores |
| Relative via rotation | Dot(q_m, k_n) depends on \`m-n\` |
| Large \`theta\` (Llama 3: 500k) | Supports longer contexts than old 10k |
| Match checkpoint \`theta\`/scale | Wrong PE ⇒ silent quality death under loaded weights |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Even \`head_dim\`; build inv-freq from \`theta\` |
| 2 | Cache cos/sin up to max seq (memory tradeoff) |
| 3 | Apply after Q/K project, before SDPA |
| 4 | Long context: Llama-3 wavelength smoothing (\`rope-scaling\` later) |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Shape unchanged; even head dim; applied to Q and K |
| **Hard** | Checkpoint \`theta\` / scale factors match |
| **Soft** | Nearby tokens score differently than far ones |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Theta / scaling | Long-context behavior |
| Absolute PE → RoPE | Modern default; don’t mix families with foreign weights |
| Skip RoPE | Model can’t reliably use order |

### 6. This course

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep05_rope.py
\`\`\`

## Next

**GQA** (\`/gqa/\`): head layout that saves KV cache.

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

**What:** **Grouped-Query Attention (GQA)** — more query heads than KV heads; groups of Q heads **share** one KV head. Still causal.

**Why:** Full MHA stores a KV pair per Q head → decode VRAM explodes with length. GQA cuts cache size with small quality hit. Llama 3.2 ~1B uses **32 Q / 8 KV**.

**How:** Choose divisible \`n_q\` / \`n_kv\`, project, RoPE, **repeat_interleave** KV to match Q, then SDPA; cache stores only \`n_kv\`.

### 1. What — shapes in this repo

[\`CausalSelfAttention\`](https://github.com/chaxus/arche/blob/main/arche/attention.py):

\`\`\`python
q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
# ... RoPE ...
k_full = new_cache.k.repeat_interleave(self.n_rep, dim=1)
v_full = new_cache.v.repeat_interleave(self.n_rep, dim=1)
attn = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=...)
\`\`\`

\`n_rep = num_heads // num_kv_heads\`. Cache appends **narrow** K/V; broadcast only for the matmul.

### 2. Why — MHA / GQA / MQA / MLA

| Layout | Idea | Cost |
| --- | --- | --- |
| **MHA** | \`n_kv = n_q\` | Richest, biggest cache |
| **GQA** | \`n_kv\` divides \`n_q\` | Open midsize default |
| **MQA** | \`n_kv = 1\` | Extreme cache save |
| **MLA** | Compressed latent KV | Frontier (DeepSeek)—later part |

Head-count change ⇒ **retrain** (or lose the checkpoint).

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Fix \`embed_dim = n_q × head_dim\` |
| 2 | Pick \`n_kv\` that divides \`n_q\` |
| 3 | Bias-free Q/K/V/O linears (Llama contract) |
| 4 | Causal mask on prefill; with cache, attend all past keys |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Valid groups; causal; weight head counts match config |
| **Soft** | Cache bytes ≈ \`n_kv/n_q\` of MHA |
| **Soft** | Quality holds vs MHA at same width |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Fewer KV heads | Decode memory / speed |
| Too few | Quality cliff |
| MLA path | Different compression story (Frontier) |

### 6. This course

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep06_gqa.py
\`\`\`

## Next

**SwiGLU** (\`/swiglu/\`): the dense FFN.

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

**What:** **SwiGLU** — gated feed-forward: \`down(silu(gate(x)) * up(x))\` with three **bias-free** linears.

**Why:** Attention mixes *across* positions; the FFN transforms *each* position. SwiGLU (not plain GELU MLP) is the dense default in Llama/Qwen; MoE experts are often the same unit.

**How:** Pick \`intermediate_dim\`, three mats, wire after the second pre-norm in the block.

### 1. What — code

[\`arche/mlp.py\`](https://github.com/chaxus/arche/blob/main/arche/mlp.py):

\`\`\`python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
\`\`\`

Gate and up expand \`D → I\`; down returns \`I → D\`. No bias—matches Llama weight files.

### 2. Why — gated vs plain MLP

| Choice | Why |
| --- | --- |
| SiLU gate × up | Soft gating; empirically strong for LMs |
| Three mats | Slightly more params than two-mat GELU MLP at same \`I\` |
| No bias | Smaller, matches open Llama/Qwen tensors |
| Position-wise | Parallelizable; pairs with residual |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Choose \`I\` (Llama 3.2 1B-style intermediate from config) |
| 2 | \`gate_proj\`, \`up_proj\`, \`down_proj\` bias=False |
| 3 | Place after \`post_attention_layernorm\` |
| 4 | MoE later: many SwiGLUs + router |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | In/out dim \`D\`; no bias; activation is SiLU on gate |
| **Soft** | Judge via full-model loss / quality, not isolated FFN |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Wider \`I\` | Capacity and FLOPs |
| GELU → SwiGLU | Needs retrain |
| MoE experts | Same brick × E, sparse fire |

### 6. This course

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep07_swiglu.py
\`\`\`

## Next

**Transformer block** (\`/block/\`): sandwich the pieces.

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

**What:** One **Transformer block** — the repeatable layer: pre-RMSNorm → attention → residual → pre-RMSNorm → SwiGLU → residual.

**Why:** Bricks only become a deep net when packaged with **residuals**. Same shape in and out so you can stack \`N\` times (Llama 3.2 ~1B: **N=16**).

**How:** Wire the sandwich once; each layer gets its own weights; stack and train (or load).

### 1. What — forward

[\`TransformerBlock\`](https://github.com/chaxus/arche/blob/main/arche/block.py):

\`\`\`python
h, new_cache = self.self_attn(self.input_layernorm(x), cache=cache, start_pos=start_pos)
x = x + h
x = x + self.mlp(self.post_attention_layernorm(x))
return x, new_cache
\`\`\`

\`\`\`mermaid
flowchart LR
  x[x] --> n1[RMSNorm]
  n1 --> attn[GQA_RoPE]
  attn --> add1[x_plus_h]
  add1 --> n2[RMSNorm]
  n2 --> ffn[SwiGLU]
  ffn --> add2[x_plus_ffn]
\`\`\`

### 2. Why — residuals + pre-norm

| Choice | Why |
| --- | --- |
| Residual add | Gradient highway; identity path if sublayer ≈0 |
| Pre-norm (not post) | Modern deep training stability |
| Attn then MLP | Llama/Qwen mainline (not parallel attn∥MLP) |
| Per-layer KV cache | Decode reuses past K/V per block |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Instantiatate norms, \`CausalSelfAttention\`, \`SwiGLU\` from config |
| 2 | Keep in/out \`embed_dim\` |
| 3 | Thread \`cache\` / \`start_pos\` for generate |
| 4 | Repeat \`N\` times in the stack |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Residuals are adds; pre-norm placement; shape \`(B,T,D)\` |
| **Hard** | Cache isolated per layer |
| **Soft** | Deep stack still trains / loads without NaNs |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Larger \`N\` | Capability and cost |
| Post → pre-norm | Enables deep from-zero training |
| Swap insides (MLA/MoE) | Frontier variants; block boundary stays |

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

**What:** The full **dense causal LM shell**: \`ids → embed → Block×N → final RMSNorm → Linear(V) → logits\`.

**Why:** One block is not a language model. The stack is what \`model(batch)\` in pretrain called—a black box until now.

**How:** Build from config (\`V,D,N,heads,…\`); optional **tie** LM head ↔ embed (small Llamas often do); init, then load or pretrain.

### 1. What — forward

[\`Llama\`](https://github.com/chaxus/arche/blob/main/arche/model.py):

\`\`\`python
x = self.embed_tokens(input_ids)
for layer, cache in zip(self.layers, caches):
    x, new_cache = layer(x, cache=cache, start_pos=start_pos)
x = self.norm(x)
return self.lm_head(x), new_caches
\`\`\`

Logits shape \`(B, T, V)\` — exactly what \`lm_loss\` consumes after shift.

### 2. Why — tying and final norm

| Choice | Why |
| --- | --- |
| Final RMSNorm | Stabilize before vocab projection |
| Tie head↔embed | Fewer params; common at ~1B |
| Untied head | More freedom at larger scales |
| Shared config object | One source of truth for HF map |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Fill \`Llama32Config\` (vocab, dims, heads, rope, …) |
| 2 | \`ModuleList\` of \`N\` blocks |
| 3 | Set \`tie_word_embeddings\` before init/load |
| 4 | Train with pretrain loop **or** \`from_pretrained\` |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Logits \`(B,T,V)\`; config ↔ weights; tying consistent |
| **Soft** | Quality after load or enough pretrain steps |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Depth / width / V | Architecture-level; retrain or migrate carefully |
| MoE swap | Replaces SwiGLU inside blocks—skeleton stays |
| Optimize modules | Don’t “tune the stack alone” without data/optim |

### 6. This course

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep09_stack.py
\`\`\`

## Next

**Load weights** (\`/weights/\`): fill the shell with Llama floats.

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

**What:** **Load** an open Llama 3.2 checkpoint into your handwritten stack via a **key map** (safetensors / HF → \`state_dict\`).

**Why:** Random weights = correct graph, nonsense text. Official floats make the same forward a language model *now*. You already saw how pretrain *writes* floats; this chapter **borrows** a finished run so generate/align work without a cluster.

**How:** Accept license → download → map keys → pair the **official tokenizer** → sanity-check shapes.

### 1. What — entry

[\`from_pretrained\`](https://github.com/chaxus/arche/blob/main/arche/load_weights.py) builds \`Llama(cfg)\` and copies HF tensors into matching names.

| Path | Role |
| --- | --- |
| **Load (this chapter)** | Example checkpoint → speak immediately |
| **Pretrain (earlier)** | How those floats get written from zero |
| **HF align (next-next)** | Proof map + formulas match HF logits |

### 2. Why — map carefully

| Risk | Effect |
| --- | --- |
| Wrong key / transpose | Silent garbage or crash |
| Wrong tokenizer | Ids point at wrong embed rows |
| Dtype / device slip | Slow or NaN |
| Partial load | “Runs” but broken layers |

### 3. How — checklist

| Step | Do |
| --- | --- |
| 1 | \`huggingface-cli login\` if gated |
| 2 | Resolve model id (course default Llama 3.2 ~1B) |
| 3 | Config → empty \`Llama\` → copy weights |
| 4 | Always use the **same** tokenizer as the checkpoint |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | No missing/unexpected keys; shapes match |
| **Hard** | Tokenizer vocabulary size = embed rows |
| **Soft** | Short generate looks like language; HF align later |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Fix key map | Unlocks all downstream demos |
| Resume your own ckpt | Same API, your floats |
| Quantized loads | Later serve chapters |

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

License, login, mirror—same class of issues as tokenizer download.

## Next

**Text generation** (\`/generate/\`): loop sample on logits.

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

**What:** **Autoregressive generation** — turn logits into text by looping: score → pick next id → append → stop.

**Why:** A trained/loaded stack only emits scores. Products differ in sampling and stop rules; they share this loop. Without it you have a classifier-shaped forward, not a chatbot.

**How:** Last-position logits → optional temperature / top-k → greedy or sample → append → halt on EOS or max length. KV cache (next part) makes the same loop fast.

### 1. What — skeleton

[\`arche/generate.py\`](https://github.com/chaxus/arche/blob/main/arche/generate.py) roughly:

\`\`\`python
# each step
logits, caches = model(ids_or_last_token, caches=caches, start_pos=...)
next_id = sample(logits[:, -1, :], temperature=..., top_k=...)  # or argmax
# append; break on EOS / max_new_tokens
\`\`\`

Prefill runs the full prompt once; decode feeds one new token at a time when cache is on.

### 2. Why — knobs ≠ weights

| Knob | Changes |
| --- | --- |
| Temperature / top-k / top-p | Style of picks—not the model floats |
| Greedy argmax | Deterministic, often dull |
| EOS / max length | UX and safety bounds |
| Chat template | How roles become ids (later device/quant/chat) |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Encode prompt with the **paired** tokenizer |
| 2 | Forward → take last time’s logits |
| 3 | Sample or argmax → append id |
| 4 | Repeat until stop; decode ids → text |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Stops; only past tokens visible; valid vocab ids |
| **Soft** | With real weights, coherent short completions |
| **Soft** | Same prompt + greedy ≈ stable across runs |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Sampling only | Behavior without retraining |
| No KV cache | Correct but O(T²) feel—fix next |
| Constrained decoding | Structured outputs |
| Bad stop rules | Runaway or clipped answers |

### 6. This course

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep11_generate.py
\`\`\`

## Next

**HF align** (\`/hf-align/\`): prove the handwritten stack matches HF.

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

**What:** **Numeric check** — same short prompt through arche vs Hugging Face Llama 3.2; compare logits (max abs diff + argmax match).

**Why:** End of Part 1 proof: key map, RMSNorm, RoPE, GQA, SwiGLU, stack layout aren’t “shape-right but wrong.” Catch bugs before serve/optim chapters hide them.

**How:** Fixed dtype; load both models; encode one prompt; forward; report. Skip cleanly if weights/login unavailable.

### 1. What — report

[\`compare_logits_to_hf\`](https://github.com/chaxus/arche/blob/main/arche/align_hf.py):

\`\`\`python
hf_logits = hf(ids).logits
arche_logits, _ = arche(ids)
max_abs = (arche_logits - hf_logits).abs().max()
argmax_match = torch.equal(arche_logits.argmax(-1), hf_logits.argmax(-1))
\`\`\`

\`run_align_or_explain\` returns \`None\` when HF weights aren’t reachable—**skip ≠ broken lesson logic**.

### 2. Why — this gate

| Failure mode | Likely cause |
| --- | --- |
| Huge \`max_abs\` | Wrong map, RoPE theta, norm, or head layout |
| Argmax mismatch | Same—plus dtype / device mismatch |
| Skip | No login / gated / offline—fix env, not formulas |

### 3. How — from zero

| Step | Do |
| --- | --- |
| 1 | Login if needed; same model id as weights chapter |
| 2 | float32 CPU compare for stability in the lab |
| 3 | Short prompt (“The cat sat”) |
| 4 | Assert within tolerance or print skip reason |

### 4. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | With weights: argmax match; \`max_abs\` under lab tol |
| **Hard** | Without weights: explicit skip, not a fake pass |
| **Soft** | Longer prompts / more positions in CI later |

### 5. Improve · whole-model impact

| Lever | Impact |
| --- | --- |
| Fail → revisit | Maps / RoPE / norms before “optimizing” |
| Pass | Safe to move to align (SFT/DPO) and serve |
| Engine CI | Same idea gates vLLM-class numeric regs |

### 6. This course

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

Part 1 closes here. Part 2 steers the base next-token skill (SFT/DPO).

## Run

\`\`\`bash
huggingface-cli login
python chapters/01-inference/ep21_hf_align.py
\`\`\`

## Fix

No weights/login → skip means “can’t download,” not “your math is wrong.”

## Next

**SFT · DPO** (\`/sft-dpo/\`): align after the base next-token skill.

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
