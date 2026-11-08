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

**What:** This course — **one path from zero toward frontier-ready**, with Llama 3.2 ~1B as the dense example.

**Why:** Jargon-first tours leave you lost. You need a finishable spine: tables → next-token train → handwritten trunk → align → serve → MoE/MLA map.

**How:** Read left-nav chapters; run each lab; mark-as-read is local-only. Repo: [github.com/chaxus/arche](https://github.com/chaxus/arche)

**Example:** Llama 3.2 ~1B = frontier dense bricks (RMSNorm, RoPE, GQA, SwiGLU, byte BPE) at a teachable scale. Official tokenizer + weights = finished pair for checks.

| Stage | In plain words |
| --- | --- |
| **0 Start** | Setup; read a chapter; run one command |
| **1 Tables + train + bricks** | Tokenizer → embedding → objective→data→optim→scale → RMSNorm…HF |
| **2 Align** | SFT / DPO |
| **3 Serve faster** | KV, long context, device/quant, continuous batch, paging |
| **4 Frontier** | MoE + MLA map, checkpoints, shards |

In-course = **runnable principles + a frontier map**. Full engines / huge clusters / multimodal = next climb *with* that map.

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

**What:** An LLM at inference **guesses the next token**; at training it learns that guess by rewriting floats.

**Why:** Chat UIs hide the loop. Weights are not magic files—they are the result of huge next-token practice (then optional align).

**How (inference):** score every vocab id → pick (argmax/sample) → append → repeat.

**How (training):** true next id is known → cross-entropy → backprop → embed rows and trunk weights move. After embedding: [\`/pretrain/\`](/pretrain/) → data → optim → scale → then dense bricks.

Three names: **token** = chunk; **embedding** = float row; **logits** = next-id scores.

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

**What:** A working Python env so every later lab can run.

**Why:** Without a venv + deps, “run the chapter” fails before any model idea appears. No GPU required for the beginner path.

**How:** Python ≥ 3.10 → venv → \`pip install -r requirements.txt\` → check script prints \`OK\`.

## Run

\`\`\`bash
git clone https://github.com/chaxus/arche.git && cd arche
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python chapters/00-start/ep00_check_setup.py
\`\`\`

No \`git\`? Download the ZIP from GitHub instead.

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

**What:** The few shell moves every lab assumes: where you are, list files, enter the repo, activate venv.

**Why:** Most early failures are “wrong directory” or “venv off,” not bad math.

**How:**

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

**What:** How this site shows code — Library (\`arche/\`) vs Lab script (\`chapters/...\`).

**Why:** You learn by running and reading, not by writing Python on day one. Knowing where to look beats memorizing APIs.

**How:** On each chapter, open the **Code** card:

1. **Library** — the module the lesson is about  
2. **Lab script** — the file you run  

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

**What:** A one-page map of the five stages and when you need Hugging Face login.

**Why:** Without the map, Part 1 feels like random bricks. Login only matters for official Llama load/align—not for teach-BPE or most shape labs.

**How:** Follow Previous/Next; keep this picture in mind:

\`\`\`mermaid
flowchart LR
  p0[0 Setup] --> p1[1 Tables_train_bricks]
  p1 --> p2[2 Align]
  p2 --> p3[3 Serve faster]
  p3 --> p4[4 Frontier]
\`\`\`

**One path:** tokenizer → embedding → **next-token train** (objective → data → optim → scale) → dense bricks → load/check Llama → align → serve → frontier map.

- **Running example:** Llama 3.2 ~1B.  
- **Core idea early:** pretrain writes embedding rows and trunk weights.  
- **HF login** for official Llama load / HF align only.

## Next

Part 1: **Overview** (\`/overview/\`).
`,
  },
];
