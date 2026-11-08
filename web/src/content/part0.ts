import type { Chapter } from './types';

/** Part 0 — absolute beginners (EN), kept short on purpose. */
export const PART0_EN: Chapter[] = [
  {
    id: 'start-welcome',
    ep: '00a',
    title: 'Welcome',
    summary: 'What this course is and how to move.',
    script: '',
    partId: 'start',
    estimatedMinutes: 2,
    markdown: `## Idea

**arche** hand-writes a **Llama 3.2–style** dense decoder (not GPT-2 archaeology).

- Left nav = chapters (Part 0 → 4). Bottom = Previous / Next.
- Mark as read stores progress in *this* browser only.
- Labs are Python files under \`chapters/\` — Part 0 first, then run them.

Repo: [github.com/chaxus/arche](https://github.com/chaxus/arche)

## Next

**What an LLM does** (\`?c=start-llm-idea\`).
`,
  },
  {
    id: 'start-llm-idea',
    ep: '00b',
    title: 'What an LLM does',
    summary: 'Next-token prediction in one loop.',
    script: '',
    partId: 'start',
    estimatedMinutes: 4,
    markdown: `## Idea

An LLM repeatedly picks the **next token** from what came before:

1. Score every token in the vocab.
2. Pick one (greedy or sampled).
3. Append it; repeat.

**Token** = text chunk (word or subword). **Embedding** = its vector. **Logits** = next-token scores.

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

**Set up your computer** (\`?c=start-setup\`).
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

**Fix:** \`No module named torch\` → venv not active or pip step skipped. Network issues → retry / PyPI mirror.

## Next

**Terminal survival** (\`?c=start-terminal\`).
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

**How to read a chapter script** (\`?c=start-reading-code\`).
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

Open \`chapters/00-start/ep00_check_setup.py\` and find:

1. Top docstring — how to run  
2. \`import\` — tools  
3. \`def main\` — steps  
4. \`print\` — output  
5. \`if __name__ == "__main__"\` — entry point  

Reading + running is enough for now.

## Next

**Map of the journey** (\`?c=start-map\`).
`,
  },
  {
    id: 'start-map',
    ep: '00f',
    title: 'Map of the journey',
    summary: 'Parts 1–4 and when HF login is needed.',
    script: '',
    partId: 'start',
    estimatedMinutes: 3,
    markdown: `## Idea

\`\`\`mermaid
flowchart LR
  p0[0 Setup] --> p1[1 Dense trunk]
  p1 --> p2[2 Faster inference]
  p2 --> p3[3 Tiny training]
  p3 --> p4[4 Frontier]
\`\`\`

- **Llama 3.2 lineage** = RMSNorm, RoPE, SwiGLU, GQA (target: 1B).
- **HF login** only for gated tokenizer/weights chapters; most demos use tiny random tensors.

## Next

Part 1: **Overview** (\`?c=overview\`).
`,
  },
];
