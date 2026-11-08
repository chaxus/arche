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

We use **Llama 3.2** ~1B as the specimen: production-like dense parts; official table+weights show a shipped pair; teaching scripts show the **from-scratch** work you do when you train your own.

**The full path (five stages):**

| Stage | In plain words | Names you will meet later |
| --- | --- | --- |
| **0 Start** | Set up; learn to read a chapter and run one command | — |
| **1 Build the dense trunk** | How tokenizer/embeds are made → hand-write RMSNorm, RoPE, GQA, SwiGLU… → load or later train → generate → HF check | dense decoder |
| **2 Train from zero** | Feel how pretrain / SFT / DPO write the weights | pretrain, SFT, DPO |
| **3 Generate & serve faster** | KV cache, long context, device/quant, continuous batch, paging | KV, RoPE scaling, paged KV… |
| **4 Frontier structures** | MoE, MLA, checkpoint map—touch the shape | MoE, MLA, shards |

**Two paths (you can interleave):**

- **Path A (specimen):** official Llama tokenizer + weights → verify the handwritten stack  
- **Path B (from zero):** teach-BPE table → build embeds → Part 2 pretrain/align → then serving  

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
    summary: 'In one loop: keep guessing the next little text chunk.',
    script: '',
    partId: 'start',
    estimatedMinutes: 4,
    markdown: `## Idea

A large language model (LLM) repeatedly guesses the **next little text chunk** from what it already wrote:

1. Score every candidate in the vocabulary.
2. Pick one (always the top score, or sample by probability).
3. Append it; repeat.

Three names you will meet: **token** = a small chunk of text; **embedding** = that chunk as a list of numbers (a vector); **logits** = scores for the next chunk.

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
    summary: 'Four from-zero stages + specimen/self-train paths; when HF login is needed.',
    script: '',
    partId: 'start',
    estimatedMinutes: 3,
    markdown: `## Idea

\`\`\`mermaid
flowchart LR
  p0[0 Setup] --> p1[1 Build trunk]
  p1 --> p2[2 Train from zero]
  p2 --> p3[3 Serve faster]
  p3 --> p4[4 Frontier structures]
\`\`\`

**Two paths:**

| Path | What you do | When |
| --- | --- | --- |
| **A Specimen** | Official tokenizer/weights + HF align | Match a shipped model |
| **B From zero** | Teach BPE, toy embeds, toy pretrain/SFT | Feel how tables and weights are written |

- **Goal:** write “how the next word appears” from zero → train weights → serve faster → frontier structure map.  
- **Specimen:** Llama 3.2 ~1B.  
- **HF login** only for path A; teach BPE and most shape labs need no login.

## Next

Part 1: **Overview** (\`/overview/\`).
`,
  },
];
