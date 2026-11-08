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

This course is **one path from zero toward frontier-ready**, with **Llama 3.2 ~1B** as the dense example you can finish on a laptop. Jargon-first tours leave you lost; here the spine is tables → next-token train → handwritten trunk → align → serve → a MoE/MLA map at the end.

Read chapters in the left nav, run each lab, and use mark-as-read locally if you like. All code lives in [github.com/chaxus/arche](https://github.com/chaxus/arche).

**Llama 3.2 ~1B** is frontier dense bricks (RMSNorm, RoPE, GQA, SwiGLU, byte BPE) at a teachable scale. The official tokenizer plus weights are the finished pair you will load for real checks—not a second curriculum.

| Stage | In plain words |
| --- | --- |
| **0 Start** | Setup; read a chapter; run one command |
| **1 Tables + train + bricks** | Tokenizer → embedding → objective→data→optim→scale → RMSNorm…HF |
| **2 Align** | SFT / DPO |
| **3 Serve faster** | KV, long context, device/quant, continuous batch, paging |
| **4 Frontier** | MoE + MLA map, checkpoints, shards |

In-course you get **runnable principles plus a frontier map**. Full engines, huge clusters, and multimodal stacks are the next climb *with* that map in hand.

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

At inference an LLM **guesses the next token**; at training it learns that guess by rewriting floats. Chat UIs hide the loop, but weights are not magic files—they are the result of huge next-token practice, then optional alignment.

**Inference:** score every vocab id → pick one (argmax or sample) → append → repeat.

**Training:** the true next id is known → cross-entropy → backprop → embedding rows and trunk weights move. After embedding comes [\`/pretrain/\`](/pretrain/) (objective → data → optim → scale), then you hand-write the dense bricks.

Three names worth locking in: **token** = text chunk; **embedding** = one float row per id; **logits** = scores for the next id.

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

Every later lab assumes a working Python environment. Without a venv and dependencies, “run the chapter” fails before any model idea shows up. The beginner path does **not** require a GPU.

Use Python ≥ 3.10, create a venv, install \`requirements.txt\`, and run the check script until it prints \`OK\`.

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

**Terminal basics** (\`/start-terminal/\`).
`,
  },
  {
    id: 'start-terminal',
    ep: '00d',
    title: 'Terminal basics',
    summary: 'cd, list files, read the last lines of a traceback.',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 5,
    markdown: `## Idea

Labs assume you can see where you are, list files, enter the repo, and activate the venv. Most early failures are “wrong directory” or “venv off,” not bad math.

On errors, read the **last** paragraph of the traceback first—that is usually the actual cause.

| Goal | macOS / Linux | Windows |
| --- | --- | --- |
| Where am I? | \`pwd\` | \`pwd\` |
| List | \`ls\` | \`dir\` |
| Enter repo | \`cd arche\` | \`cd arche\` |

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

This site splits code into the **library** (\`arche/\`) and the **lab script** (\`chapters/...\`). You learn by running and reading, not by writing Python on day one. Knowing where to look beats memorizing APIs.

On each chapter, open the **Code** card: **Library** is the module the lesson is about; **Lab script** is the file you run.

In a script, skim in order: docstring → \`import\` → \`def main\` → \`print\` → \`if __name__\`.

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

Here is the five-stage map and when Hugging Face login actually matters. Without it, Part 1 can feel like random bricks. Login is only for loading or aligning **official Llama**—not for teach-BPE or most shape labs.

Follow Previous/Next and keep this picture in mind:

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
