import type { Chapter } from './types';
import { PART0_EN } from './part0';

export const CHAPTERS_EN: Chapter[] = [
  ...PART0_EN,
{
    id: 'overview',
    ep: '01',
    title: 'Overview',
    summary: 'Course map: Llama 3.2 ~1B spine from tables through frontier.',
    script: 'chapters/01-inference/ep01_overview.py',
    partId: 'trunk',
    estimatedMinutes: 12,
    markdown: `## Idea

This chapter is the **course map**: one path from zero toward frontier-ready, with **Llama 3.2 ~1B** as the running dense example. You will print its config early so sizes stay real, not hand-wavy.

If the example recipe drifts, later “how industry does it” will not line up. The spine is tables → next-token train → bricks → align → serve → MoE/MLA—each brick gets a place on that line before the details matter.

In practice: nail the example → grow tokenizer and embed → train next-token (objective → data → optim → scale) → hand-write RMSNorm…stack → load, generate, HF checks → SFT/DPO → serve → frontier primers.

### The path

| Stage | Plain words |
| --- | --- |
| **0 Start** | Setup; read a chapter; run one command |
| **1 Tables + train + bricks** | Tokenizer → embedding → **pretrain×4** → RMSNorm…stack → weights → generate → HF |
| **2 Align** | SFT / DPO |
| **3 Serve faster** | KV, RoPE scaling, device/quant/chat, continuous batch, paging |
| **4 Frontier** | MoE → MLA → ckpt → shards |

\`\`\`mermaid
flowchart LR
  tok[tokenizer] --> emb[embedding]
  emb --> train[pretrain_loop]
  train --> bricks[dense_bricks]
  bricks --> align[SFT_DPO]
  align --> serve[serve]
  serve --> tip[MoE_MLA]
\`\`\`

You hand-write a **Llama 3.2-style dense decoder**. Official Llama files are that recipe’s finished pair—not a second curriculum.

### Why Llama 3.2 ~1B

Same brick set as larger Llama 3 and many Qwen-class dense stacks: **RMSNorm + RoPE + GQA + SwiGLU + byte BPE**. At ~1B, vocab, width, depth, and head counts stay countable and the weights are downloadable—**frontier-shaped architecture at a teachable scale**.

| Pick | Meaning |
| --- | --- |
| Frontier-shaped bricks | Names match current open dense + serving code |
| Teachable scale | Finishable from zero |
| Open finished pair | Tokenizer + checkpoint for real checks |
| Dense first | MoE/MLA later as maps |

This is not “smartest chatbot” and not “train 70B at home.”

### Where DeepSeek fits

| | **Llama 3.2 (example)** | **DeepSeek-V2/V3** |
| --- | --- | --- |
| Capacity | Dense FFN | **MoE** top-k experts |
| KV | **GQA** | **MLA** latents |
| In-course | Parts 1–3 | Part 4 primers |

Same Transformer family; different upgrades. You build the dense trunk first.

### Config you will keep seeing

The lab ([\`ep01_overview.py\`](https://github.com/chaxus/arche/blob/main/chapters/01-inference/ep01_overview.py)) prints [\`llama32_1b()\`](https://github.com/chaxus/arche/blob/main/arche/config.py):

| Field | ~1B idea |
| --- | --- |
| \`vocab_size\` | 128256 |
| \`num_layers\` | 16 |
| \`embed_dim\` | 2048 |
| \`num_heads\` / \`num_kv_heads\` | 32 / 8 (GQA) |
| \`rope_theta\` | 500000 |

### Check yourself

- Config is self-consistent; bricks match the claimed recipe.
- From zero still works: tables trainable, net stackable, weights loadable.
- Docs and obtainable weights exist; MoE/MLA primers have somewhere to attach.

### Changing the example

Swap the example (e.g. Qwen) and you retarget tokenizer, embeds, and blocks. Wider or deeper means retrain. This course keeps 1B and uses primers for larger ideas.

### In this repo

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

Prints sizes—not a chatbot yet.

## Run

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

## Next

**Tokenizer** (\`/tokenizer/\`): grow the first table from zero.

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

The **tokenizer table** is cut rules plus a vocab: each text piece maps to an integer id and back. It is the first table you choose or train when building from zero.

Everything downstream—embeddings, pretraining cost, inference—locks to this id space. Swap the table mid-flight and you are roughly rebuilding embeds and running a heavy retrain.

Open decoders usually grow the table as: corpus mix → pretokenizer → byte base → BPE merges → specials → freeze and version → then embed and pretrain. In this course you get a tiny teach-BPE to *see* growth, plus Llama’s finished table for the running example.

### Cut and number

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

### Growing the table from zero

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

### What makes a good table

**Must-haves**

- **Reversible:** encode→decode recovers normal text.  
- **Coverage:** any byte string representable.  
- **Stable:** same text → same ids.  
- **Train-time identity:** same table for pretrain, finetune, and infer.

**Tradeoffs** (there is no single score):

| Concern | Plain meaning | Impact on the whole model |
| --- | --- | --- |
| **Compression / fertility** | fewer tokens better; fertility = pieces/word | longer effective context; cheaper train/infer |
| **Parity** | similar meaning shouldn’t explode across languages | unfair ⇒ short effective context, expensive, undertrained |
| **Domain fit** | code/math/CJK not shredded | vertical ceiling |
| **Structure** | indent, JSON, URLs, numbers | tools & code |
| **Vocab size** | often 32K–200K+ | shorter seqs vs **embed params ∝ V×D** and fatter softmax |
| **Special design** | clear chat/tool boundaries | alignment & agents |
| **Downstream** | loss/evals under same trunk | one final referee |

“Good” balances compression, parity, size, and downstream—not one leaderboard number.

### Improving the table · whole-model impact

| Lever | How | Whole-model impact |
| --- | --- | --- |
| **Corpus mix** | more code/CJK/math | those domains compress & learn better; others may regress |
| **Pretokenizer** | digit/CJK-friendly regex | often beats blindly growing V |
| **Larger V** | more merges | shorter sequences; fatter embed/LM head; more VRAM |
| **Append specials** | grow table tail; keep old ids | **expand embedding rows** and keep training |
| **Change algorithm family** | BPE ↔ Unigram… | **new table**; rebuild embeds; retrain |
| **Extend inherited table** | append domain pieces | cheaper than full reset |

Frontier-from-zero practice: freeze the tokenizer after data goals are clear → set \`vocab_size\` → embed → pretrain. Mid-flight table swaps are extremely expensive.

### What open models do

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

### In this course

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

The **embedding** is a \`V×D\` table: each token id **looks up** one row of \`D\` floats (\`gather\` / \`nn.Embedding\`). Neural nets only add and multiply tensors; ids are labels with no geometry. Without float rows, attention and FFN have nothing to eat—and **pretrain writes those rows**.

After you freeze the tokenizer, set \`V\` and \`D\` (and tying), random-init the table, and let next-token training rewrite the rows you actually see. Or load official floats together with the paired tokenizer.

### How an id becomes floats

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

**Why floats at all?** Attention and FFN only eat float tensors. Training nudges rows so co-occurring tokens get useful starting faces; renumbering the tokenizer breaks everything unless row index stays locked to tokenizer id.

One line: **tokenizer numbers tokens; embedding gives each number a trainable vector the rest of the net can compute on.** Bad table ⇒ later layers train on noise; wrong pairing with the tokenizer ⇒ the pipeline is junk.

At first rows are **random** (almost no meaning). Next-token training backprops through wrong guesses and **rewrites rows that showed up**—frequent tokens move most. Results land in the checkpoint; inference loads and gathers. This course can also load Llama’s already-trained rows.

The playground on this page is exactly “gather by id.”

\`\`\`mermaid
flowchart LR
  ids[tokenizer ids] --> gather[lookup rows]
  gather --> floats[T×D floats]
  floats --> blocks[later layers eat floats only]
\`\`\`

### Building the table from zero

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

### What counts as a good table

**Check yourself**

- **Shape:** \`(V, D)\` matches config and tokenizer; \`T\` ids → \`T\` rows.  
- **Pairing:** row \`i\` ↔ tokenizer id \`i\`.  
- **Healthy values:** no NaN/Inf; RMSNorm can steady scales.  
- **Contract:** this stack uses RoPE later; no absolute PE on embeds.

You rarely get a solo “embedding score.” Judges are downstream loss and same-architecture compares under a fixed setup—fitting the pretrain objective, dim budget, rare-token updates, and tie/untie choices all show up there.

### Improving embeddings · whole-model impact

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

### What open models do

| Model / line | How embeddings are done | Notes |
| --- | --- | --- |
| **Llama 3 / 3.2** | lookup; small sizes often **tie** LM head; **no** absolute PE on embeds | **course example**; RoPE for position |
| **Qwen** | lookup + RoPE; V from their tokenizer | strict pairing |
| **DeepSeek** | lookup; frontier focus on MLA/MoE etc. | multilingual tokenizer makes rows “worth more” |
| **GPT line (public)** | larger V → larger embed; generation bumps rebuild embeds | \`o200k\` isn’t a light patch on \`cl100k\` embeds |
| **BERT era** | token + segment + **absolute position** embeds | unlike modern decoder “pure lookup + RoPE” |
| **MoE LLMs** | usually still a shared lookup; sparsity in FFN experts | embed isn’t the MoE headline, but V still costs |

Consensus: embeddings are a **learnable lookup** factory-paired with the tokenizer; improving them ≈ better pretrain/tokenizer; open checkpoints store the matrix in safetensors—you write the same when training from zero.

### In this course

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

**Causal language modeling** (next-token pretraining) is the self-supervised loop that turns raw text into useful weights. Given all tokens to the left, the model scores every vocabulary id for “what comes next,” and training raises the probability on the true next id. That objective writes embedding rows and almost the entire trunk—without it you only have a random shell, or you load someone else’s floats (as in the Llama example later). In one line: tokenize text → embed → forward → **shift** so position \`t\` predicts id \`t+1\` → cross-entropy → backward → optimizer step, repeated on huge corpora.

Industry uses the same names: nanoGPT’s \`X,Y\` windows, Hugging Face “causal LM,” Llama 3 / DeepSeek “pretrain then align.”

### The objective

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

### Why this design

| Choice | Rationale |
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

### Shift, loss, and one step

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

At random init, loss ≈ \`log(V)\` (uniform over vocab).

### Check yourself

- Shift is exactly one token; causality holds (no future leak).
- Gradients reach embedding rows and trunk; no NaNs.
- Loss trends down over a sensible window; samples get less random.
- At scale, the recipe matches published pretrain practice (later chapters).

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| More / better tokens | Main capability ceiling |
| Correct shift / mask | Silent wrong objective if broken |
| LR / clip / accum | Finishability (next chapter) |
| Under-train | Weak embeds **and** deep layers |

Up next: **data windows**, **optimizers**, and **frontier scale recipes**—this chapter stays focused on the objective plus one step.

### Industry pointers

- Raschka FAQ: constructing input/target windows  
- nanoGPT \`train.py\`: readable full loop  
- HF LLM Course ch7.6: causal LM from scratch  
- Llama 3 / DeepSeek reports: same family of objective, different scale & architecture  

### This course

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

Pretraining needs tensors the loss can actually score: **\`(batch, time)\` id matrices** built from raw text. The path is document → tokenize → pack or cut fixed windows → batch → (optional) pad and mask. The objective from the previous chapter is fixed; **data shape** decides what signal each step sees. Bad windows—padding counted as labels, leaked future context, sloppy packing—quietly waste or corrupt training. Keep **labels aligned with the same shift rule** as \`lm_loss\`, whether you shift in the collator or inside the model.

### What goes into a batch

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

### Design pressures

| Pressure | Consequence |
| --- | --- |
| GPU likes dense \`(B,T)\` | Prefer packing over tiny ragged sequences |
| Context length \`T\` | Caps how far “next token” can look back |
| Pad tokens | Must **not** train as if they were real next words (mask or ignore_index) |
| Doc boundaries | Blind concat can invent fake “next tokens” across unrelated docs unless you mask |

### Toy windows in this repo

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

### Check yourself

- Every supervised position has a real next id (or is masked out).
- Label/input shift matches your loss implementation—no double shift.
- Token utilization is high (little pure padding).
- The domain mix matches what you want the model to learn.

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| Cleaner / more tokens | Stronger base model |
| Longer \`T\` | Better long-range next-token skill (cost ↑) |
| Packing quality | Throughput and effective updates/day |
| Domain mix | What the model becomes “fluent” at |

### This course

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

Once the loss is right, **the optimizer side** of each step decides whether weights move in a stable way. AdamW, learning rate (and schedules), gradient clipping, and gradient accumulation are the usual knobs. Correct CE with a bad optimizer still gives NaNs, divergence, or a flat loss—past toy scale, **how you step** is what makes a run finish. The skeleton is: \`zero_grad\` → forward → loss → \`backward\` → (optional clip) → \`optimizer.step\` → (optional lr schedule; accumulate over micro-batches when VRAM forces small \`B\`).

### Knobs you will touch

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

No schedule or clip yet—on purpose. Production adds them around the same skeleton.

### Why each piece exists

| Choice | Rationale |
| --- | --- |
| AdamW over plain SGD | Sparse-ish token grads + many params; adaptive scales help |
| Weight decay | Mild regularization; AdamW’s form matches LLM practice |
| Warmup | Early CE is huge (\`~log V\`); blunt lr shreds init |
| Clip | One bad batch shouldn’t blow FP math |
| Accum | Fit long \`T\` / big models on limited VRAM while keeping effective batch |

### Gradient accumulation sketch

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

### Check yourself

- No NaN/Inf in loss or grads after clip.
- \`zero_grad\` runs once per **optimizer** step—not forgotten under accumulation.
- Loss trends down over a sensible window.
- When you scale up, lr schedule matches a known recipe.

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| lr / warmup | Trainability of deep stacks + embeds |
| effective batch (B×accum×world) | Noise vs stability of updates |
| clip threshold | Spike survival vs under-updating |
| Wrong decay | Over-shrink or under-regularize weights |

### This course

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

The next-token loop you already know is the same one frontier labs run—they stretch it with **more tokens, staged recipes, and parallelism**, not a different objective. Llama 3.2 ~1B and trillion-token runs share **CE + one step**; they differ in data volume, lr schedules, cluster math, and sometimes **architecture** (dense vs MoE). This chapter is a map so “toy loss went down” does not feel like graduation. Read published recipes as extensions of pretrain, pretrain-data, and pretrain-optim: same loss, harder data, optim, and engineering.

### Axes that change with scale

| Axis | Toy (this course) | Frontier-shaped |
| --- | --- | --- |
| **Tokens** | A few random batches | Trillions of real tokens |
| **Steps / stages** | One flat loop | Multi-stage LR, data mixes, context lengthening |
| **Batch** | Tiny \`B,T\` | Huge effective batch via accum + data parallel |
| **Model** | Tiny dense \`Llama\` | Dense 1B–70B+ or MoE (DeepSeek) with MLA/experts |
| **Hardware** | One process | Megatron/FSDP/pipeline-style sharding |

Llama 3-style public reports: long pretrain on curated mixes, then shorter high-quality anneal / long-context phases—still next-token at heart.

DeepSeek-V2/V3: **same causal LM family**, different **block** (MLA + MoE)—see Frontier chapters; training still CE + big-data optim.

### Common illusions after ep18

| Illusion | Reality |
| --- | --- |
| “Loss fell on random ids” | Proves the **loop**, not language skill |
| “Copy AdamW defaults” | Defaults need warmup/schedule/clip at scale |
| “Bigger model only” | Data quality × tokens often dominate |
| “SFT will fix a bad base” | Align edits style; weak pretrain stays weak |

### Reading a recipe in four lines

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

### Check yourself

- You can name which axis a paper changed (data, optim, systems, arch).
- MoE **architecture** is not a different **loss**.
- You expect more engineering before “reproduce Llama.”

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| Token budget | Primary capability driver |
| Staged data / LR | Stability + late quality |
| Parallel efficiency | Whether the budget is usable |
| Arch (GQA/MoE/MLA) | Cost per token & memory—not a new objective |

### This course

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

**RMSNorm** keeps each token’s activations on a stable scale: divide by root-mean-square, then multiply a learnable \`γ\`. There is no mean subtraction and no bias. Deep next-token stacks blow up or collapse without that steadying step; Llama-style **pre-norm** (normalize before attention and MLP) is what makes depth trainable from scratch. In practice you place \`RMSNorm(D)\` at each pre-norm site, initialize \`γ ≈ 1\`, and train it with the rest of the net.

### Formula and code

\\[
\\mathrm{RMSNorm}(x)=\\frac{x}{\\sqrt{\\mathrm{mean}(x^2)+\\varepsilon}}\\odot\\gamma
\\]

Repo ([\`arche/norms.py\`](https://github.com/chaxus/arche/blob/main/arche/norms.py)):

\`\`\`python
variance = x_f.pow(2).mean(dim=-1, keepdim=True)
x_normed = x_f * torch.rsqrt(variance + self.eps)
return (self.weight.float() * x_normed).to(orig_dtype)
\`\`\`

Math runs in float32, then casts back—the same stability trick HF Llama uses.

### Compared with LayerNorm and placement

| Choice | Rationale |
| --- | --- |
| No mean center | Slightly cheaper; Llama/Qwen recipe |
| Learnable \`γ\` only | Scale per channel after unit-RMS |
| **Pre-norm** | Stabilize *inputs* to attn/MLP so depth survives |
| After embed, before bricks | First “steady the ruler” once vectors exist |

GPT-2-era teaching often used LayerNorm + post-norm. This course’s example is **RMSNorm + pre-norm**.

### Wiring it in

| Step | Do |
| --- | --- |
| 1 | \`RMSNorm(D)\` before attention and before SwiGLU (see block) |
| 2 | Final RMSNorm before LM head (see stack) |
| 3 | Match \`ε\` to the checkpoint (\`norm_eps\`) |
| 4 | Train / load; don’t swap LN↔RMSN under foreign weights |

### Check yourself

- Output shape matches input; \`ε\` is set; pre-norm sites match the recipe.
- With \`γ=1\`, per-token RMS ≈ 1; fewer NaNs in a deep forward.
- Full-model loss and HF align are the real judge.

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| LN → RMSN | Architecture change ⇒ retrain |
| Pre vs post | Depth trainability |
| \`γ\` / lr | Couples to convergence |

### This course

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

**RoPE** (Rotary Position Embedding) rotates query and key vectors by token position so **relative distance** shows up in the attention dot product. Embeddings say *which* token; attention still needs *where*. Absolute position tables on the embedding are an older pattern; modern dense Llama, Qwen, and DeepSeek apply RoPE on Q and K. Pick \`rope_theta\`, cache cos/sin, combine with \`rotate_half\` on Q/K; long-context scaling is a later knob.

### Apply step in code

Repo ([\`arche/rope.py\`](https://github.com/chaxus/arche/blob/main/arche/rope.py)):

\`\`\`python
def apply_rotary_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
\`\`\`

\`build_rope_cache\` precomputes cos/sin for \`0 .. seq_len-1\`. Attention slices by \`start_pos\` when decoding with cache.

### Design choices

| Choice | Rationale |
| --- | --- |
| Rotate Q/K only | V stays content; positions live in scores |
| Relative via rotation | Dot(q_m, k_n) depends on \`m-n\` |
| Large \`theta\` (Llama 3: 500k) | Supports longer contexts than old 10k |
| Match checkpoint \`theta\`/scale | Wrong PE ⇒ silent quality death under loaded weights |

### Wiring from scratch

| Step | Do |
| --- | --- |
| 1 | Even \`head_dim\`; build inv-freq from \`theta\` |
| 2 | Cache cos/sin up to max seq (memory tradeoff) |
| 3 | Apply after Q/K project, before SDPA |
| 4 | Long context: Llama-3 wavelength smoothing (\`rope-scaling\` later) |

### Check yourself

- Shapes unchanged; even head dim; RoPE on both Q and K.
- Checkpoint \`theta\` and scale factors match.
- Nearby tokens score differently than distant ones.

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| Theta / scaling | Long-context behavior |
| Absolute PE → RoPE | Modern default; don’t mix families with foreign weights |
| Skip RoPE | Model can’t reliably use order |

### This course

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

**Grouped-Query Attention (GQA)** uses more query heads than KV heads: groups of Q heads **share** one K/V head. It is still fully causal attention. Full multi-head attention stores a separate KV pair per Q head, so decode memory grows fast with sequence length; GQA shrinks the KV cache with a usually small quality cost. Llama 3.2 ~1B runs **32 Q / 8 KV**. Pick divisible \`n_q\` and \`n_kv\`, project, apply RoPE, **repeat_interleave** KV to line up with Q, then SDPA; the cache stores only \`n_kv\` heads.

### Shapes in this repo

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

### MHA, GQA, MQA, and MLA

| Layout | Idea | Cost |
| --- | --- | --- |
| **MHA** | \`n_kv = n_q\` | Richest, biggest cache |
| **GQA** | \`n_kv\` divides \`n_q\` | Open midsize default |
| **MQA** | \`n_kv = 1\` | Extreme cache save |
| **MLA** | Compressed latent KV | Frontier (DeepSeek)—later part |

Changing head counts ⇒ **retrain** (or lose the checkpoint).

### Wiring from scratch

| Step | Do |
| --- | --- |
| 1 | Fix \`embed_dim = n_q × head_dim\` |
| 2 | Pick \`n_kv\` that divides \`n_q\` |
| 3 | Bias-free Q/K/V/O linears (Llama contract) |
| 4 | Causal mask on prefill; with cache, attend all past keys |

### Check yourself

- Valid groups; causal; weight head counts match config.
- Cache bytes ≈ \`n_kv/n_q\` of MHA.
- Quality holds vs MHA at the same width.

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| Fewer KV heads | Decode memory / speed |
| Too few | Quality cliff |
| MLA path | Different compression story (Frontier) |

### This course

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

**SwiGLU** is the gated feed-forward block: \`down(silu(gate(x)) * up(x))\` with three **bias-free** linear layers. Attention mixes *across* positions; the FFN transforms *each* position on its own. SwiGLU—not a plain GELU MLP—is the dense default in Llama and Qwen; MoE experts are often the same brick. Pick \`intermediate_dim\`, three projection matrices, and place the module after the second pre-norm in each block.

### Code in the repo

[\`arche/mlp.py\`](https://github.com/chaxus/arche/blob/main/arche/mlp.py):

\`\`\`python
class SwiGLU(nn.Module):
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
\`\`\`

Gate and up expand \`D → I\`; down returns \`I → D\`. No bias—matches Llama weight files.

### Gated FFN vs plain MLP

| Choice | Rationale |
| --- | --- |
| SiLU gate × up | Soft gating; empirically strong for LMs |
| Three mats | Slightly more params than two-mat GELU MLP at same \`I\` |
| No bias | Smaller, matches open Llama/Qwen tensors |
| Position-wise | Parallelizable; pairs with residual |

### Wiring from scratch

| Step | Do |
| --- | --- |
| 1 | Choose \`I\` (Llama 3.2 1B-style intermediate from config) |
| 2 | \`gate_proj\`, \`up_proj\`, \`down_proj\` bias=False |
| 3 | Place after \`post_attention_layernorm\` |
| 4 | MoE later: many SwiGLUs + router |

### Check yourself

- In/out dim \`D\`; no bias; SiLU on the gate branch.
- Judge via full-model loss / quality, not an isolated FFN.

### Levers for the whole model

| Lever | Impact |
| --- | --- |
| Wider \`I\` | Capacity and FLOPs |
| GELU → SwiGLU | Needs retrain |
| MoE experts | Same brick × E, sparse fire |

### This course

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

One **Transformer block** is the repeatable layer in a decoder stack: pre-RMSNorm → attention → residual → pre-RMSNorm → SwiGLU → residual. Residuals are what let you stack many identical layers without the gradient dying—the input and output stay the same shape so you can repeat the block \`N\` times (Llama 3.2 ~1B uses **N=16**). In arche you wire this sandwich once in [\`block.py\`](https://github.com/chaxus/arche/blob/main/arche/block.py); each layer owns its own weights, and the stack chapter repeats the block.

### Forward pass

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

### Residuals and pre-norm

| Choice | Why it matters |
| --- | --- |
| Residual add | Gradient highway; identity path if sublayer ≈0 |
| Pre-norm (not post) | Modern deep training stability |
| Attn then MLP | Llama/Qwen mainline (not parallel attn∥MLP) |
| Per-layer KV cache | Decode reuses past K/V per block |

### Build from scratch

| Step | Do |
| --- | --- |
| 1 | Instantiate norms, \`CausalSelfAttention\`, \`SwiGLU\` from config |
| 2 | Keep in/out \`embed_dim\` |
| 3 | Thread \`cache\` / \`start_pos\` for generate |
| 4 | Repeat \`N\` times in the stack |

### Check yourself

| Must hold | |
| --- | --- |
| Residuals are adds; pre-norm placement; shape \`(B,T,D)\` | |
| Cache isolated per layer | |
| Worth aiming for | Deep stack still trains / loads without NaNs |

### Tuning the whole model

| Lever | Impact |
| --- | --- |
| Larger \`N\` | Capability and cost |
| Post → pre-norm | Enables deep from-zero training |
| Swap insides (MLA/MoE) | Frontier variants; block boundary stays |

### This course

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

The **dense causal LM shell** wires embedding, \`Block×N\`, final RMSNorm, and the vocabulary head: \`ids → embed → Block×N → final RMSNorm → Linear(V) → logits\`. A single block is not yet a language model—this stack is what pretrain’s \`model(batch)\` has been calling from the outside. Build from config (\`V,D,N,heads,…\`), optionally **tie** LM head and embeddings (common on small Llamas), then init, load, or pretrain.

### Forward pass

[\`Llama\`](https://github.com/chaxus/arche/blob/main/arche/model.py):

\`\`\`python
x = self.embed_tokens(input_ids)
for layer, cache in zip(self.layers, caches):
    x, new_cache = layer(x, cache=cache, start_pos=start_pos)
x = self.norm(x)
return self.lm_head(x), new_caches
\`\`\`

Logits shape \`(B, T, V)\` — exactly what \`lm_loss\` consumes after shift.

### Tying and final norm

| Choice | Why it matters |
| --- | --- |
| Final RMSNorm | Stabilize before vocab projection |
| Tie head↔embed | Fewer params; common at ~1B |
| Untied head | More freedom at larger scales |
| Shared config object | One source of truth for HF map |

### Build from scratch

| Step | Do |
| --- | --- |
| 1 | Fill \`Llama32Config\` (vocab, dims, heads, rope, …) |
| 2 | \`ModuleList\` of \`N\` blocks |
| 3 | Set \`tie_word_embeddings\` before init/load |
| 4 | Train with pretrain loop **or** \`from_pretrained\` |

### Check yourself

| Must hold | Logits \`(B,T,V)\`; config ↔ weights; tying consistent |
| Worth aiming for | Quality after load or enough pretrain steps |

### Tuning the whole model

| Lever | Impact |
| --- | --- |
| Depth / width / V | Architecture-level; retrain or migrate carefully |
| MoE swap | Replaces SwiGLU inside blocks—skeleton stays |
| Optimize modules | Don’t “tune the stack alone” without data/optim |

### This course

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

This chapter **loads** an open Llama 3.2 checkpoint into your handwritten stack: safetensors / HF names land in arche’s \`state_dict\` through a **key map**. Random weights mean a correct graph with nonsense text—official floats turn the same forward into a language model right away. You already saw how pretrain *writes* those floats; here you **borrow** a finished run so generate and HF align work without your own cluster. Accept the license, download, map keys, pair the **official tokenizer**, and sanity-check shapes.

### Entry point

[\`from_pretrained\`](https://github.com/chaxus/arche/blob/main/arche/load_weights.py) builds \`Llama(cfg)\` and copies HF tensors into matching names.

| Path | Role |
| --- | --- |
| **Load (this chapter)** | Example checkpoint → speak immediately |
| **Pretrain (earlier)** | How those floats get written from zero |
| **HF align (next-next)** | Proof map + formulas match HF logits |

### Mapping pitfalls

| Risk | Effect |
| --- | --- |
| Wrong key / transpose | Silent garbage or crash |
| Wrong tokenizer | Ids point at wrong embed rows |
| Dtype / device slip | Slow or NaN |
| Partial load | “Runs” but broken layers |

### Checklist

| Step | Do |
| --- | --- |
| 1 | \`huggingface-cli login\` if gated |
| 2 | Resolve model id (course default Llama 3.2 ~1B) |
| 3 | Config → empty \`Llama\` → copy weights |
| 4 | Always use the **same** tokenizer as the checkpoint |

### Check yourself

| Must hold | No missing/unexpected keys; shapes match |
| Must hold | Tokenizer vocabulary size = embed rows |
| Worth aiming for | Short generate looks like language; HF align later |

### Tuning the whole model

| Lever | Impact |
| --- | --- |
| Fix key map | Unlocks all downstream demos |
| Resume your own ckpt | Same API, your floats |
| Quantized loads | Later serve chapters |

### This course

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

**Autoregressive generation** is the loop that turns logits into text: score → pick the next token id → append → stop. A trained or loaded stack only emits scores until you wrap it in this loop; products differ in sampling and stop rules, but they all share the same shape. Take last-position logits, optional temperature / top-k, greedy or sample, halt on EOS or max length—KV cache (next chapter) makes the same loop fast at long outputs.

### Loop skeleton

[\`arche/generate.py\`](https://github.com/chaxus/arche/blob/main/arche/generate.py) roughly:

\`\`\`python
# each step
logits, caches = model(ids_or_last_token, caches=caches, start_pos=...)
next_id = sample(logits[:, -1, :], temperature=..., top_k=...)  # or argmax
# append; break on EOS / max_new_tokens
\`\`\`

Prefill runs the full prompt once; decode feeds one new token at a time when cache is on.

### Knobs are not weights

| Knob | Changes |
| --- | --- |
| Temperature / top-k / top-p | Style of picks—not the model floats |
| Greedy argmax | Deterministic, often dull |
| EOS / max length | UX and safety bounds |
| Chat template | How roles become ids (later device/quant/chat) |

### Wire it up

| Step | Do |
| --- | --- |
| 1 | Encode prompt with the **paired** tokenizer |
| 2 | Forward → take last time’s logits |
| 3 | Sample or argmax → append id |
| 4 | Repeat until stop; decode ids → text |

### Check yourself

| Must hold | Stops; only past tokens visible; valid vocab ids |
| Worth aiming for | With real weights, coherent short completions |
| Worth aiming for | Same prompt + greedy ≈ stable across runs |

### Tuning behavior without retraining

| Lever | Impact |
| --- | --- |
| Sampling only | Behavior without retraining |
| No KV cache | Correct but O(T²) feel—fix next |
| Constrained decoding | Structured outputs |
| Bad stop rules | Runaway or clipped answers |

### This course

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

**HF align** runs the same short prompt through arche and Hugging Face Llama 3.2, then compares logits—max absolute diff and whether argmax matches. That’s the Part 1 proof your key map, RMSNorm, RoPE, GQA, SwiGLU, and stack layout aren’t merely shape-correct. Fix dtype, load both models, encode one prompt, forward, and report; skip cleanly when weights or login aren’t available.

### What the script reports

[\`compare_logits_to_hf\`](https://github.com/chaxus/arche/blob/main/arche/align_hf.py):

\`\`\`python
hf_logits = hf(ids).logits
arche_logits, _ = arche(ids)
max_abs = (arche_logits - hf_logits).abs().max()
argmax_match = torch.equal(arche_logits.argmax(-1), hf_logits.argmax(-1))
\`\`\`

\`run_align_or_explain\` returns \`None\` when HF weights aren’t reachable—**skip ≠ broken lesson logic**.

### When numbers disagree

| Failure mode | Likely cause |
| --- | --- |
| Huge \`max_abs\` | Wrong map, RoPE theta, norm, or head layout |
| Argmax mismatch | Same—plus dtype / device mismatch |
| Skip | No login / gated / offline—fix env, not formulas |

### Run the check

| Step | Do |
| --- | --- |
| 1 | Login if needed; same model id as weights chapter |
| 2 | float32 CPU compare for stability in the lab |
| 3 | Short prompt (“The cat sat”) |
| 4 | Assert within tolerance or print skip reason |

### Check yourself

| Must hold | With weights: argmax match; \`max_abs\` under lab tol |
| Must hold | Without weights: explicit skip, not a fake pass |
| Worth aiming for | Longer prompts / more positions in CI later |

### If you fail or pass

| Lever | Impact |
| --- | --- |
| Fail → revisit | Maps / RoPE / norms before “optimizing” |
| Pass | Safe to move to align (SFT/DPO) and serve |
| Engine CI | Same idea gates vLLM-class numeric regs |

### This course

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

**KV cache** keeps each layer’s Keys and Values after prefill, then **appends** one new step at decode instead of recomputing the full past every token. A correct generate loop without cache recomputes attention over growing history on every step—fine for demos, unusable at product lengths. Production decoders all cache KV; GQA, MLA, and paging shrink what you store on top of that.

### Cache structure

[\`KVCache\`](https://github.com/chaxus/arche/blob/main/arche/cache.py):

\`\`\`python
@dataclass
class KVCache:
    k: torch.Tensor  # (batch, kv_heads, seq, head_dim)
    v: torch.Tensor

    def append(self, k, v):
        return KVCache(k=torch.cat([self.k, k], dim=-2), v=torch.cat([self.v, v], dim=-2))
\`\`\`

Attention ([\`CausalSelfAttention\`](https://github.com/chaxus/arche/blob/main/arche/attention.py)) RoPE-applies the new Q/K, merges into cache, then SDPA over **full** cached K/V.

### Cost shape

| Mode | Work per new token |
| --- | --- |
| No cache | Re-embed + full stack over all past ids |
| With cache | One new position’s Q/K/V + attend to cache |
| Memory | Grows with \`layers × n_kv × seq × dim\` |

GQA already cut \`n_kv\`; paging (later) fights fragmentation across users.

### Wire decode

| Step | Do |
| --- | --- |
| 1 | One cache slot per layer |
| 2 | Prefill: \`cache=None\`, write full prompt K/V |
| 3 | Decode: pass cache + \`start_pos\`, append one step |
| 4 | Never mix layers’ caches |

### Check yourself

| Must hold | Matches full recompute numerically (toy check) |
| Must hold | No layer mixups; seq dim only grows |
| Worth aiming for | Memory fits target context × batch |

### Tuning memory

| Lever | Impact |
| --- | --- |
| GQA / MLA / KV-quant | Smaller cache |
| Paging | Multi-user packing |
| No cache | Toy lengths only |

### This course

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

**RoPE scaling** adjusts rotary frequencies (Llama 3-style wavelength smoothing) so positions beyond the original training window stay usable. Short-context pretrain plus naive long indices makes rotations wrap badly and quality collapses far out. Long-context products need a **recipe extension**, not “just set max_seq bigger.” Match checkpoint \`theta\` and scale factors, optionally continue-train on long data, and eval both short and long.

### Code hook

[\`llama3_inv_freq\`](https://github.com/chaxus/arche/blob/main/arche/rope.py) / \`build_rope_cache(..., scale_factor=..., low_freq_factor=..., high_freq_factor=..., original_max_position=...)\`.

When \`scale_factor == 1.0\`, plain RoPE. Otherwise wavelength bands are scaled/smoothed the HF Llama-3 way—**same math the 128k-class configs advertise**.

### Length is not free

| Illusion | Reality |
| --- | --- |
| Stretch indices only | High-freq bands misbehave |
| Bigger window free | KV memory ↑ with length |
| Scaling = long knowledge | Still need data / continued train for hard long tasks |

### Apply the recipe

| Step | Do |
| --- | --- |
| 1 | Read official rope scaling fields from the ckpt config |
| 2 | Build cos/sin with those factors |
| 3 | Keep short-context sanity checks |
| 4 | Long-bench / needle tests before claiming 128k |

### Check yourself

| Must hold | Factors match the checkpoint |
| Worth aiming for | Near context still stable; far context usable |
| Worth aiming for | Long-bench acceptable for the product bar |

### Whole-model impact

| Lever | Impact |
| --- | --- |
| Wrong scaling | Breaks long (and maybe short) |
| YaRN / NTK / … | Other open recipes—don’t mix under Llama weights |
| Longer windows | Always pay in KV |

### This course

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

**Mixture-of-Experts (MoE)** swaps the dense SwiGLU for many experts plus a router; each token only runs **top-k** of them. You get huge parameter capacity while per-token compute stays near the dense case. DeepSeek-V3’s open story is largely MoE for capacity and MLA for KV; Mixtral is another open MoE. This course stayed on dense Llama 3.2 first so you can read the FFN before the sparse map.

Pick \`E\` and \`top_k\`, wire experts and a gate, add load-balance loss in serious training, and plan how experts land in checkpoints.

### Forward pass

[\`MoESwiGLU\`](https://github.com/chaxus/arche/blob/main/arche/moe.py):

\`\`\`python
logits = self.gate(x)
probs = F.softmax(logits, dim=-1)
top_probs, top_idx = probs.topk(self.top_k, dim=-1)
# weighted sum of selected SwiGLU experts
\`\`\`

The decoder spine stays the same; only the MLP slot changes.

### Dense vs sparse

| | Dense SwiGLU | MoE |
| --- | --- | --- |
| Params | One FFN | \`E\` × FFN |
| Per-token FLOPs | Full FFN | ≈ \`top_k\` experts |
| Failure mode | — | Expert collapse without balancing |

### From zero

| Step | Do |
| --- | --- |
| 1 | Keep attention / norms; swap MLP → MoE |
| 2 | Choose \`E\`, \`top_k\` (e.g. Mixtral-like top-2) |
| 3 | Add load-balance loss in serious training |
| 4 | Plan expert-parallel / sharded loads |

### Check yourself

- Routes are valid; merge weights sum sanely.
- Expert load stays roughly balanced; quality/capacity trade feels OK.
- Checkpoint maps cover every expert.

### Levers

| Lever | Impact |
| --- | --- |
| More \`E\` / \`k\` | Capacity vs compute |
| Balance loss | Fights collapse |
| Expert parallel | Multi-GPU engineering cost |

### Industry map

| Line | Role |
| --- | --- |
| **DeepSeek-V3** | Large MoE + MLA (next) |
| **Mixtral** | Top-2 SwiGLU experts |
| **Llama 3.2 ~1B (course)** | Dense first |

### This course

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

## Run

\`\`\`bash
python chapters/01-inference/ep14_moe.py
\`\`\`

## Next

**MLA** (\`/mla/\`): compress KV the DeepSeek way.

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

**Multi-head Latent Attention (MLA)** compresses KV into low-rank **latents** plus a small RoPE stream, instead of caching full per-head K/V. After GQA, long-context cache can still explode; DeepSeek-V2/V3 push MLA here while Llama/Qwen dense mainlines stay on GQA. Read with MoE: capacity vs KV.

This lesson is a **shape primer**, not a bit-exact DeepSeek port. Pick a latent rank, project, cache latent + RoPE key, expand when attending.

### Cache shape

[\`MLACache\` / \`MultiHeadLatentAttention\`](https://github.com/chaxus/arche/blob/main/arche/mla.py):

\`\`\`python
@dataclass
class MLACache:
    kv_latent: torch.Tensor  # (batch, seq, kv_lora_rank)
    k_rope: torch.Tensor     # (batch, seq, rope_dim)
\`\`\`

When rank ≪ heads×dim, this is much smaller than full \`(batch, kv_heads, seq, head_dim)\` KV.

### Versus GQA

| | GQA (Llama path) | MLA (DeepSeek path) |
| --- | --- | --- |
| Store | Shared KV heads | Latent + tiny RoPE |
| Goal | Cut heads | Cut rank further |
| Swap cost | Architecture-level either way |

### From zero

| Step | Do |
| --- | --- |
| 1 | Choose \`kv_lora_rank\`, optional \`q_lora_rank\` |
| 2 | Decouple RoPE dims from content latent |
| 3 | Cache latents across decode steps |
| 4 | Compare memory vs GQA at the same quality bar |

### Check yourself

- Shapes stay consistent; cache appends along seq.
- Cache is smaller; quality near full KV for your product bar.
- Don’t drop MLA under Llama GQA weights.

### Levers

| Lever | Impact |
| --- | --- |
| Smaller latent | Save memory / risk quality |
| + paging | Same serve tricks as GQA stacks |
| Mix under wrong weights | Silent wrong attention |

### Industry map

| Line | Attention / KV |
| --- | --- |
| **DeepSeek-V2 / V3** | MLA (+ MoE body) |
| **Llama 3 / Qwen dense** | GQA mainline |

### This course

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

This chapter is shipping glue: **device** placement, a teaching **INT8** shrink, and **chat templates** that turn roles into token ids. A correct dense stack still fails users if it won’t run on their hardware, doesn’t fit memory, or formats chat wrong—these aren’t new layers, they’re the last mile. Probe device, optional quant recipe, apply **this model’s** template, then wire into generate and KV.

### Pieces in the repo

| Piece | Role in repo |
| --- | --- |
| Device fallback | CPU/CUDA/MPS pick for demos |
| INT8 toy | Teaching shrink—not GPTQ/AWQ/FP8 production |
| Chat template | [\`apply_chat_template\` / \`encode_chat\`](https://github.com/chaxus/arche/blob/main/arche/chat.py) |

Wrong template ⇒ model “ignores” system/user structure even with perfect weights.

### Product constraints

| Pressure | Response |
| --- | --- |
| No GPU | CPU path must work (slow OK for labs) |
| VRAM tight | Quant / smaller dtype |
| Multi-turn UX | Official chat template + specials |
| Multi-backend | Engineering cost, not math |

### Ship a demo

| Step | Do |
| --- | --- |
| 1 | Probe available device |
| 2 | Load weights in a runnable dtype |
| 3 | Format messages with the **paired** template |
| 4 | Generate with KV as usual |

### Check yourself

| Must hold | Fallback runs; template matches tokenizer/model |
| Worth aiming for | Quant quality acceptable for the demo bar |
| Worth aiming for | Roles render clearly in the prompt string/ids |

### Beyond the toy path

| Lever | Impact |
| --- | --- |
| Harder quant | Smaller/faster/riskier |
| Wrong template | Junk dialogues |
| Prod quant (GPTQ/AWQ/FP8) | Engine territory (vLLM/TGI/llama.cpp) |

### This course

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

**Continuous batching** is the serving **scheduler** shape: many users share decode steps; finished jobs retire and new ones admit mid-flight. A single-user generate loop wastes GPU when one sequence ends early. Engines like vLLM, TGI, and SGLang keep the device busy by stepping only **active** requests together. Active set → pad actives → sample → retire on EOS/max → repeat—the teaching scheduler is not a full paged-attention engine.

### Scheduler loop

[\`generate_continuous\`](https://github.com/chaxus/arche/blob/main/arche/batching.py):

\`\`\`python
while any(not item.done for item in live):
    active = [item for item in live if not item.done]
    # pad active rows → model(batch) → sample per row → retire finished
\`\`\`

Each request carries its own temperature / top-k / max tokens. No weight change—pure scheduling.

### Throughput picture

| Single loop | Continuous batch |
| --- | --- |
| One user owns the step | Many share a step |
| Idle after short finish | Retire + admit keeps GPU full |
| Simple | Needs careful KV layout in prod |

### Build the toy scheduler

| Step | Do |
| --- | --- |
| 1 | Represent each job (\`DecodeRequest\`) |
| 2 | Each step: build tensor from **active only** |
| 3 | Retire on EOS / length |
| 4 | Later: paged KV so memory matches the schedule |

### Check yourself

| Must hold | Finished jobs stop; no cross-talk between rows |
| Worth aiming for | Higher tokens/s on mixed-length traffic |
| Worth aiming for | Steadier tail latency under load |

### Serving ceiling

| Lever | Impact |
| --- | --- |
| Better schedule | More throughput, same weights |
| Coupled to KV layout | Prod needs paging / blocks |
| Ignoring batching | Leaves GPU on the table |

### This course

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

**Alignment after pretrain**: SFT teaches the model to answer like your demos; DPO (and RLHF-family methods) nudge **chosen vs rejected** preferences without a full RL loop in this lesson. Next-token pretrain gives you a capable base LM—it does not by itself follow instructions or match product taste. Alignment steers an existing net; it does not invent the base skill. Flow: instruction data → mask-prompt SFT from a base ckpt → preference pairs → DPO → regression evals.

### SFT loss

[\`sft_loss\`](https://github.com/chaxus/arche/blob/main/arche/train/sft.py): same shift as pretrain, but **only response tokens** get CE:

\`\`\`python
# shift_logits vs shift_labels as usual
mask[i, start:] = True  # start ≈ prompt_len - 1
return losses[mask].mean()
\`\`\`

Prompt tokens are context, not supervised targets—otherwise the model “learns to rewrite the user.”

### DPO sketch

[\`dpo_loss\`](https://github.com/chaxus/arche/blob/main/arche/train/dpo.py): policy vs frozen **reference**; prefer chosen completion over rejected (β-scaled).

| Stage | Data | Objective |
| --- | --- | --- |
| **SFT** | prompt → good reply | CE on reply span |
| **DPO** | prompt + chosen/rejected | Prefer chosen logprob margin vs ref |
| **RLHF** (map only) | reward model + PPO-like | Not coded here |

### Order and risks

| Choice | Why it matters |
| --- | --- |
| After pretrain | Need a next-token-capable base |
| Mask prompts | Don’t burn loss on user text |
| Ref model in DPO | Anchor so policy doesn’t drift unbound |
| Data > blind steps | Bad demos teach bad habits |

Over-align can dull base capability; often paired with new specials / chat templates.

### Run a toy training step

| Step | Do |
| --- | --- |
| 1 | Start from base (or instruct) checkpoint |
| 2 | Build prompt/response ids + \`prompt_lens\` |
| 3 | \`train_sft_step\` / \`train_dpo_step\` toys |
| 4 | Eval following + regressions before ship |

### Check yourself

| Must hold | Correct masks/shapes; ref frozen in DPO |
| Worth aiming for | Better following / safety; limited capability loss |
| Worth aiming for | Template + tokenizer stay paired |

### This course

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

Frontier MoE/MLA only helps if you can speak their **checkpoint names and shapes**—save and load with a key map, including Mixtral-like \`w1/w2/w3\` expert tensors. Structure on paper is useless if tensors won’t round-trip; a bad map leaves silent random experts.

Map open names into your modules, use [\`save_checkpoint\` / \`load_checkpoint\`](https://github.com/chaxus/arche/blob/main/arche/checkpoint.py), and fail loudly on missing keys.

### Round-trip

Teaching ckpts write state + meta, reload into a matching module graph, and assert alignment. Every expert key must be present—partial MoE loads can “run” while one expert is noise.

### Naming debt

| Risk | Effect |
| --- | --- |
| HF vs local name drift | Missing keys / wrong slots |
| Expert index off-by-one | One dead expert forever |
| Strict=False abuse | Hides map bugs |

### From zero

| Step | Do |
| --- | --- |
| 1 | List open hub key patterns (Mixtral / DeepSeek docs) |
| 2 | Build an explicit map into your modules |
| 3 | Round-trip a tiny synthetic ckpt first |
| 4 | Only then touch multi-GB downloads |

### Check yourself

- Round-trip \`allclose\`; no missing experts.
- Unexpected keys fail or are explicitly ignored.
- Hooks are ready for real hubs.

### Levers

| Lever | Impact |
| --- | --- |
| Consistent naming | Cuts converter debt |
| Bad maps | Silent quality holes |
| Converters as standard | Same story as dense HF maps |

### This course

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

Open frontier weights usually arrive as **\`index.json\` + shards**, not one file. If you can’t assemble shards, you can’t load Llama/Qwen/DeepSeek/Mixtral-class hubs; bad concat silently corrupts tensors.

Use an HF-style index, partition keys, write shard files, and aggregate on load—toy size, same protocol.

### API

[\`write_sharded_state\` / \`load_sharded_state\`](https://github.com/chaxus/arche/blob/main/arche/shards.py):

\`\`\`python
write_sharded_state(directory, state_dict, shard_size=...)
state = load_sharded_state(directory)  # rebuild full dict via index
\`\`\`

Synthetic pack/unpack plus \`allclose\` teaches the protocol **without** multi-tens-of-GB downloads.

### Why hubs shard

| Single file | Sharded |
| --- | --- |
| Simple demos | Hub default for big models |
| One download stream | Parallel shard fetch |
| Peak RAM = full model | Still careful—assemble wisely |

### From zero

| Step | Do |
| --- | --- |
| 1 | Emit index mapping tensor name → shard file |
| 2 | Split by size/count policy |
| 3 | Load: read index, pull only needed shards |
| 4 | Fail clearly on missing pieces |

### Check yourself

- Index matches shards; reload aligns.
- Missing shard/key errors are loud.
- Shard policy balances download vs peak RAM.

### Levers

| Lever | Impact |
| --- | --- |
| Shard policy | Parallelism and memory peaks |
| Bad concat | Silent bad weights |
| safetensors + index | Industry dominant pair |

### This course

\`\`\`bash
python chapters/02-training/ep22_sharded_frontier.py
\`\`\`

Path complete—revisit weak chapters, or re-run HF align with weights.

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

Two serving accelerators share this lesson: **paged KV** (block table + shared page pool) fights memory fragmentation, and **CUDA Graph** captures a fixed-shape decode step to cut launch overhead. Continuous batching without paging wastes VRAM on ragged lengths; eager CUDA launches tax every tiny decode. Together they point at vLLM-class engines—here you get a teaching stand-in, not a drop-in replacement. Logical KV → alloc/free pages via block table → \`gather_kv\` for attention; optionally capture graphs when decode shapes stay stable.

### Paged KV

[\`PagedKVPool\` / \`BlockTable\` / \`gather_kv\`](https://github.com/chaxus/arche/blob/main/arche/paged_kv.py):

\`\`\`python
# append one step into the next slot; alloc a new page when a page fills
append_kv(pool, table, k, v)
# materialize contiguous KVCache for SDPA
cache = gather_kv(pool, table)
\`\`\`

Physical pages are shared; each sequence owns a list of page ids.

### CUDA Graph (optional)

[\`capture_module_step\` / \`generate_with_optional_graph\`](https://github.com/chaxus/arche/blob/main/arche/cuda_graphs.py): capture a module step when CUDA is available; replay instead of re-launching. Needs **stable shapes**—why engines care about batching policy.

### Toward real engines

| Piece | Pain it hits |
| --- | --- |
| Paging | Fragmentation + multi-user packing |
| Graphs | Per-step CPU launch overhead |
| Neither changes weights | Raises the serving ceiling only |

### Wire it up

| Step | Do |
| --- | --- |
| 1 | Continuous batch mental model first |
| 2 | Pool + per-seq block table |
| 3 | Gather before attention (teaching path) |
| 4 | Optional graph on decode when shapes freeze |

### Check yourself

| Must hold | Gather matches logical cache; tables in-bounds |
| Worth aiming for | Less fragmentation / launch tax; higher throughput |
| Worth aiming for | Clear skip when no CUDA |

### This course

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

Part 3 ends; Part 4 maps MoE/MLA frontier structures.

## Run

\`\`\`bash
python chapters/01-inference/ep23_paged_cuda_graph.py
\`\`\`

## Next

**MoE** (\`/moe/\`): frontier capacity via sparse experts.

`,
  },

];
