import type { Chapter } from './types';

/** Part 0 — absolute beginners (EN), kept short on purpose. */
export const PART0_EN: Chapter[] = [
  {
    id: 'start-welcome',
    ep: '00a',
    title: 'Welcome',
    summary: 'From zero: how an LLM is built, sped up, trained, and what the tip adds.',
    script: '',
    partId: 'start',
    estimatedMinutes: 5,
    markdown: `## Idea

**What you will learn:** treat the LLM as something you **train from zero toward frontier-ready**—how tokenizer/embedding tables are made, what “good” means, how to improve them and what that does to the whole model; hand-write the trunk until it runs; then speed, a bit of training, and a map of MoE/MLA-style tips. No jargon required up front.

**One path, one example.** You build from zero on a **frontier dense recipe**: **Llama 3.2 ~1B**—the small open sibling of Meta’s Llama 3 line. Same architecture family used at larger scales: RMSNorm, RoPE, GQA, SwiGLU, byte BPE—what today’s open dense stacks (and many closed ones) still ship, not a GPT-2 classroom toy. We pick the **~1B** scale so those bricks stay teachable and downloadable; the **architecture** is the point. Official tokenizer + weights = that recipe’s finished pair for checking your stack.

**The full path (five stages):**

| Stage | In plain words | Names you will meet later |
| --- | --- | --- |
| **0 Start** | Set up; learn to read a chapter and run one command | — |
| **1 Tables + next-token train + dense bricks** | Tokenizer → embedding → **objective → data → optim → scale** → RMSNorm…stack → load/generate → HF check | dense decoder, pretrain |
| **2 Align** | SFT / DPO after the base next-token skill | SFT, DPO |
| **3 Generate & serve faster** | KV cache, long context, device/quant, continuous batch, paging | KV, RoPE scaling, paged KV… |
| **4 Frontier structures** | MoE + MLA map (DeepSeek-V3-shaped open stack), checkpoints | MoE, MLA, shards |

**Afterward you can roughly:** run the labs; open similar open-source code and know which layer you are in; enter papers or repos like vLLM with a map. In-course = **runnable principles + a frontier map**. Full industrial engines, huge clusters, multimodal—are the next climb *with* that map, not from zero.

Site: left nav; Previous / Next; Mark as read is local-only. Repo: [github.com/chaxus/arche](https://github.com/chaxus/arche)

## Next

**What an LLM does** (\`/start-llm-idea/\`).
`,
  },
  {
    id: 'start-llm-idea',
    ep: '00b',
    title: 'What an LLM does',
    summary: 'Inference guesses the next token; training teaches that guess by rewriting floats.',
    script: '',
    partId: 'start',
    estimatedMinutes: 5,
    markdown: `## Idea

At **inference**, an LLM repeatedly guesses the **next little text chunk** from what it already wrote:

1. Score every candidate in the vocabulary.
2. Pick one (argmax or sample).
3. Append it; repeat.

At **training** (the important half), the dataset already knows the true next chunk. The model still scores every id; you measure the miss (**cross-entropy**), then **backprop** so embedding rows and later weights move. Huge text × many steps = how random floats become a language model. After embedding the spine splits that loop: [\`/pretrain/\`](/pretrain/) (objective) → data → optim → scale primer → then dense bricks.

Three names: **token** = chunk; **embedding** = float row for an id; **logits** = next-chunk scores.

\`\`\`mermaid
flowchart LR
  text[Text] --> tok[Tokens]
  tok --> vec[Vectors]
  vec --> stack[Layers]
  stack --> scores[Scores]
  scores --> pick[Next token]
  pick --> text
\`\`\`

## Next

**Set up your computer** (\`/start-setup/\`).
`,
  },
  {
    id: 'start-setup',
    ep: '00c',
    title: 'Set up your computer',
    summary: 'Python 3.10+, venv, install deps, run the check.',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 15,
    markdown: `## Idea

Need Python ≥ 3.10 and a venv. No GPU required.

## Run

\`\`\`bash
git clone https://github.com/chaxus/arche.git && cd arche
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python chapters/00-start/ep00_check_setup.py
\`\`\`

Expect a final \`OK\` line. No \`git\`? Download the ZIP from GitHub instead.

## Fix

\`No module named torch\` → venv not active or pip step skipped. Network issues → retry / PyPI mirror.

## Next

**Terminal survival** (\`/start-terminal/\`).
`,
  },
  {
    id: 'start-terminal',
    ep: '00d',
    title: 'Terminal survival',
    summary: 'cd, list files, read the last lines of a traceback.',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 5,
    markdown: `## Idea

| Goal | macOS / Linux | Windows |
| --- | --- | --- |
| Where am I? | \`pwd\` | \`pwd\` |
| List | \`ls\` | \`dir\` |
| Enter repo | \`cd arche\` | \`cd arche\` |

On errors, read the **last** traceback paragraph first.

## Run

\`\`\`bash
cd arche && source .venv/bin/activate
python chapters/00-start/ep00_check_setup.py
python chapters/01-inference/ep01_overview.py   # optional smoke
\`\`\`

## Next

**How to read a chapter script** (\`/start-reading-code/\`).
`,
  },
  {
    id: 'start-reading-code',
    ep: '00e',
    title: 'How to read a chapter script',
    summary: 'Spot import, main, print — no need to write Python yet.',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 4,
    markdown: `## Idea

This page shows the lab script below. In any later chapter, use the **Code** card:

1. **Library** — the \`arche/\` module you are learning  
2. **Lab script** — the \`chapters/...\` file you run  

In a script, spot: docstring → \`import\` → \`def main\` → \`print\` → \`if __name__\`.

## Next

**Map of the journey** (\`/start-map/\`).
`,
  },
  {
    id: 'start-map',
    ep: '00f',
    title: 'Map of the journey',
    summary: 'One from-zero spine with Llama as the example; when HF login is needed.',
    script: '',
    partId: 'start',
    estimatedMinutes: 3,
    markdown: `## Idea

\`\`\`mermaid
flowchart LR
  p0[0 Setup] --> p1[1 Tables_train_bricks]
  p1 --> p2[2 Align]
  p2 --> p3[3 Serve faster]
  p3 --> p4[4 Frontier]
\`\`\`

**One path:** tokenizer → embedding → **next-token train** → dense bricks → load/check Llama → align → serve → frontier map.

- **Running example:** Llama 3.2 ~1B.  
- **Core idea early:** next-token pretrain (objective → data → optim → scale) is how embedding rows and trunk weights get written.  
- **HF login** when you load or align against official Llama files; teach-BPE and most shape labs need no login.

## Next

Part 1: **Overview** (\`/overview/\`).
`,
  },
];
