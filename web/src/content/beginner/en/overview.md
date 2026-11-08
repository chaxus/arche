## Idea

**What:** The **course map** — one path from zero toward frontier-ready, with **Llama 3.2 ~1B** as the running dense example and a printed config to nail sizes.

**Why:** Wrong example recipe ⇒ later “how industry does it” won’t match. You need one spine (tables → next-token train → bricks → align → serve → MoE/MLA) before any single brick makes sense.

**How:** Fix the example → grow tokenizer/embed → train next-token (objective→data→optim→scale) → hand-write RMSNorm…stack → load/generate/HF align → SFT/DPO → serve → frontier primers.

### 1. What — the path

| Stage | Plain words |
| --- | --- |
| **0 Start** | Setup; read a chapter; run one command |
| **1 Tables + train + bricks** | Tokenizer → embedding → **pretrain×4** → RMSNorm…stack → weights → generate → HF |
| **2 Align** | SFT / DPO |
| **3 Serve faster** | KV, RoPE scaling, device/quant/chat, continuous batch, paging |
| **4 Frontier** | MoE → MLA → ckpt → shards |

```mermaid
flowchart LR
  tok[tokenizer] --> emb[embedding]
  emb --> train[pretrain_loop]
  train --> bricks[dense_bricks]
  bricks --> align[SFT_DPO]
  align --> serve[serve]
  serve --> tip[MoE_MLA]
```

You hand-write a **Llama 3.2-style dense decoder**. Official Llama files are that recipe’s finished pair—not a second curriculum.

### 2. Why — Llama 3.2 ~1B

**Frontier dense architecture, teachable scale.**

Same brick set as larger Llama 3 / many Qwen-class dense stacks: **RMSNorm + RoPE + GQA + SwiGLU + byte BPE**. ~1B keeps V/D/depth/heads countable and downloadable.

| Pick | Meaning |
| --- | --- |
| Frontier-shaped bricks | Names match current open dense + serving code |
| Teachable scale | Finishable from zero |
| Open finished pair | Tokenizer + checkpoint for real checks |
| Dense first | MoE/MLA later as maps |

Not “smartest chatbot,” not “train 70B at home.”

### 3. Why — where DeepSeek sits

| | **Llama 3.2 (example)** | **DeepSeek-V2/V3** |
| --- | --- | --- |
| Capacity | Dense FFN | **MoE** top-k experts |
| KV | **GQA** | **MLA** latents |
| In-course | Parts 1–3 | Part 4 primers |

Same Transformer family; different upgrades. Dense trunk first.

### 4. How — config you will keep seeing

Lab ([`ep01_overview.py`](https://github.com/chaxus/arche/blob/main/chapters/01-inference/ep01_overview.py)) prints [`llama32_1b()`](https://github.com/chaxus/arche/blob/main/arche/config.py):

| Field | ~1B idea |
| --- | --- |
| `vocab_size` | 128256 |
| `num_layers` | 16 |
| `embed_dim` | 2048 |
| `num_heads` / `num_kv_heads` | 32 / 8 (GQA) |
| `rope_theta` | 500000 |

### 5. What “good” means

| Kind | Check |
| --- | --- |
| **Hard** | Config self-consistent; bricks match the claimed recipe |
| **Hard** | Supports from-zero: tables trainable, net stackable, weights loadable |
| **Soft** | Docs + obtainable weights; room for MoE/MLA primers |

### 6. Improve · whole-model impact

Swap example (e.g. Qwen) ⇒ retarget tokenizer/embeds/blocks. Wider/deeper ⇒ retrain. This course keeps 1B and uses primers for larger ideas.

### 7. This course

```bash
python chapters/01-inference/ep01_overview.py
```

Prints sizes—not a chatbot yet.

## Run

```bash
python chapters/01-inference/ep01_overview.py
```

## Next

**Tokenizer** (`/tokenizer/`): grow the first table from zero.
