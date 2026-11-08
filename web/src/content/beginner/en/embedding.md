## Idea

The **embedding** is a `V×D` table: each token id **looks up** one row of `D` floats (`gather` / `nn.Embedding`). Neural nets only add and multiply tensors; ids are labels with no geometry. Without float rows, attention and FFN have nothing to eat—and **pretrain writes those rows**.

After you freeze the tokenizer, set `V` and `D` (and tying), random-init the table, and let next-token training rewrite the rows you actually see. Or load official floats together with the paired tokenizer.

### How an id becomes floats

Think **locker wall**, not a dictionary gloss:

1. Tokenizer says: this text became ids `7, 3, 7`
2. There are `V` lockers (vocab size); each holds exactly **`D` floats** (`D=2048` in Llama 3.2 1B)
3. **Id → floats = open that locker and copy the row out** (`gather` / `nn.Embedding`)  
   - id `7` → row 7’s `D` numbers  
   - id `3` → row 3  
   - the same id **always** hits the same row (random at init; training rewrites the numbers)

Tiny fake example (`D=3`):

| id | embedding row |
| --- | --- |
| 3 | `0.2, -0.1, 0.5` |
| 7 | `-0.3, 0.8, 0.1` |

Ids `[7, 3, 7]` → three rows: row 7, then row 3, then row 7 again. **No formula turns 7 into floats**—it is a lookup.

**Why floats at all?** Attention and FFN only eat float tensors. Training nudges rows so co-occurring tokens get useful starting faces; renumbering the tokenizer breaks everything unless row index stays locked to tokenizer id.

One line: **tokenizer numbers tokens; embedding gives each number a trainable vector the rest of the net can compute on.** Bad table ⇒ later layers train on noise; wrong pairing with the tokenizer ⇒ the pipeline is junk.

At first rows are **random** (almost no meaning). Next-token training backprops through wrong guesses and **rewrites rows that showed up**—frequent tokens move most. Results land in the checkpoint; inference loads and gathers. This course can also load Llama’s already-trained rows.

The playground on this page is exactly “gather by id.”

```mermaid
flowchart LR
  ids[tokenizer ids] --> gather[lookup rows]
  gather --> floats[T×D floats]
  floats --> blocks[later layers eat floats only]
```

### Building the table from zero

**Not** hand-filled glosses, **not** another BPE. BPE assigns ids; embedding stores the float rows—mostly **written by pretrain**.

| Step | What you do | Note |
| --- | --- | --- |
| **1. Lock V** | `V = tokenizer size` (incl. specials) | same factory as previous chapter |
| **2. Choose D** | e.g. **2048** for Llama 3.2 1B | architecture hyperparameter |
| **3. Build module** | `nn.Embedding(V, D)` | random init |
| **4. Tie or not** | small models often tie embed ↔ lm_head | larger sometimes untie |
| **5. Pretrain** | next-token CE; grads update **seen rows** | grows with the whole net |
| **6. Save** | `embed_tokens.weight` in checkpoint | infer = load + gather |
| **7. Changes** | new specials → **grow rows + train**; new tokenizer → **rebuild** | costs differ by orders of magnitude |

Init is an empty locker wall; **training writes the numbers**. This lesson’s toy table shows gather + one update; real tables come from official weights or your pretrain.

### What counts as a good table

**Check yourself**

- **Shape:** `(V, D)` matches config and tokenizer; `T` ids → `T` rows.  
- **Pairing:** row `i` ↔ tokenizer id `i`.  
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

From-zero order: freeze tokenizer → set `V,D` & tying → random embed + pretrain → append rows only when protocols need new specials. **Never** mix Llama embeds with a Qwen tokenizer.

### What open models do

| Model / line | How embeddings are done | Notes |
| --- | --- | --- |
| **Llama 3 / 3.2** | lookup; small sizes often **tie** LM head; **no** absolute PE on embeds | **course example**; RoPE for position |
| **Qwen** | lookup + RoPE; V from their tokenizer | strict pairing |
| **DeepSeek** | lookup; frontier focus on MLA/MoE etc. | multilingual tokenizer makes rows “worth more” |
| **GPT line (public)** | larger V → larger embed; generation bumps rebuild embeds | `o200k` isn’t a light patch on `cl100k` embeds |
| **BERT era** | token + segment + **absolute position** embeds | unlike modern decoder “pure lookup + RoPE” |
| **MoE LLMs** | usually still a shared lookup; sparsity in FFN experts | embed isn’t the MoE headline, but V still costs |

Consensus: embeddings are a **learnable lookup** factory-paired with the tokenizer; improving them ≈ better pretrain/tokenizer; open checkpoints store the matrix in safetensors—you write the same when training from zero.

### In this course

```python
self.weight = nn.Embedding(vocab_size, embed_dim)
return self.weight(token_ids)
```

```bash
python chapters/01-inference/ep03_embedding.py
```

Expect shape, gather, mean |Δ| after one step, finiteness. Real Llama 1B: `V≈128256`, `D=2048` in the official checkpoint.

## Run

```bash
python chapters/01-inference/ep03_embedding.py
```

## Next

**Pretrain** (`/pretrain/`): the next-token loop that **writes** those float rows (and the rest of the trunk).
