import type { Chapter } from './types';

/** Part 0 — absolute beginners (EN). */
export const PART0_EN: Chapter[] = [
  {
    id: 'start-welcome',
    ep: '00a',
    title: 'Welcome',
    summary: 'What arche is, how this site works, and how long the path takes.',
    script: '',
    partId: 'start',
    estimatedMinutes: 8,
    markdown: `## In one sentence

You will learn how a modern language model works by reading plain explanations and running short Python scripts—without needing to know Python first.

## Before you start

- Time: about 8 minutes of reading.
- You need a laptop, internet, and willingness to copy-paste commands later.
- No prior ML or coding class required.

## The idea (plain language)

**arche** (ἀρχή) means *first principle*. The project hand-writes the important pieces of a **current** open model family (Llama 3.2 style)—not an old GPT-2 museum tour.

This **website** is the classroom. The **Python files** under \`chapters/\` are lab sheets you run in a terminal. You mark chapters as read with the button at the top; that only saves on your browser.

Use the language switcher anytime (English / 中文). Use Previous / Next at the bottom to walk in order.

## In the code

Nothing to run yet. Repo: [github.com/chaxus/arche](https://github.com/chaxus/arche).

## Do this now

1. Skim the left navigation: you should see **Part 0 — Start here** first.
2. Click **Mark as read** when this page makes sense.
3. Go to the next chapter with the **Next** button.

## If something breaks

- If the page looks empty, refresh once; your progress is in local browser storage.
- If a link opens GitHub and asks you to sign in, you can still read public files without an account.

## Don't worry about yet

Model papers, GPUs, math notation, or “training from scratch on the internet.” We stay on a teaching path.

## Next

Continue to **What an LLM does** (\`?c=start-llm-idea\`).
`,
  },
  {
    id: 'start-llm-idea',
    ep: '00b',
    title: 'What an LLM does',
    summary: 'Next-token prediction in plain language—no formulas yet.',
    script: '',
    partId: 'start',
    estimatedMinutes: 12,
    markdown: `## In one sentence

A large language model (LLM) repeatedly guesses the **next piece of text**, using what came before.

## Before you start

- Read Welcome (\`start-welcome\`).
- Time: ~12 minutes.

## The idea (plain language)

Imagine a very patient autocomplete.

1. You type (or the system starts with) some text.
2. The model scores every word/piece it knows: “how likely is each next piece?”
3. It picks one (sometimes the top score, sometimes a random draw among good options).
4. That piece is appended. Go back to step 2.

Three words we will use a lot:

- **Token** — a chunk of text the model treats as one unit (sometimes a word, sometimes part of a word).
- **Embedding** — a list of numbers that stands for a token inside the model.
- **Logits / scores** — the raw preferences for “what token comes next.”

You do **not** need the formulas yet. Remember the loop: *read past → score next → append → repeat*.

\`\`\`mermaid
flowchart LR
  text[Text] --> tok[Tokens]
  tok --> vec[Vectors]
  vec --> stack[Layers]
  stack --> scores[Next-token scores]
  scores --> pick[Pick one token]
  pick --> text
\`\`\`

## In the code

Later chapters implement each box. For now, just keep the picture.

## Do this now

Say out loud (or write): “An LLM predicts the next token from the previous ones.” If that sentence feels true, you are ready.

## If something breaks

- If “token” still feels vague: think “LEGO brick of text,” not “subway token.”
- If diagrams do not render, the text above still stands alone.

## Don't worry about yet

Training data, RLHF, agents, or chat apps. Those sit on top of this loop.

## Next

**Set up your computer** (\`?c=start-setup\`).
`,
  },
  {
    id: 'start-setup',
    ep: '00c',
    title: 'Set up your computer',
    summary: 'Install Python, clone the repo, create a venv, install requirements.',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 25,
    markdown: `## In one sentence

You will install Python 3.10+, download this project, create an isolated environment, and install libraries.

## Before you start

- Time: 15–40 minutes depending on download speed.
- macOS, Windows, or Linux is fine. A GPU is **not** required.

## The idea (plain language)

Python is the language of the lab scripts. A **venv** (virtual environment) is a private toolbox so arche’s libraries do not fight other projects on your machine.

## In the code

After setup you will run:

\`\`\`bash
python chapters/00-start/ep00_check_setup.py
\`\`\`

It should print \`OK — continue...\`.

## Do this now

### 1) Install Python 3.10 or newer

- **macOS:** install from [python.org](https://www.python.org/downloads/) or \`brew install python\`
- **Windows:** install from [python.org](https://www.python.org/downloads/) and tick **Add python.exe to PATH**
- **Linux:** use your package manager (e.g. \`sudo apt install python3 python3-venv python3-pip\`)

Check:

\`\`\`bash
python3 --version
\`\`\`

(Windows may use \`python\` instead of \`python3\`.)

### 2) Open a terminal

- **macOS:** Terminal.app or iTerm
- **Windows:** Windows Terminal or PowerShell
- **Linux:** your usual terminal

### 3) Clone the repo

\`\`\`bash
git clone https://github.com/chaxus/arche.git
cd arche
\`\`\`

No git? Download the ZIP from GitHub, unzip, then \`cd\` into the folder.

### 4) Create and activate a venv

macOS / Linux:

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
\`\`\`

Windows (PowerShell):

\`\`\`powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
\`\`\`

Your prompt should show \`(.venv)\`.

### 5) Install packages

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 6) Confidence check

\`\`\`bash
python chapters/00-start/ep00_check_setup.py
\`\`\`

## If something breaks

- \`python3: command not found\` → install Python, reopen the terminal.
- Windows execution policy blocks Activate → run PowerShell as yourself and follow Microsoft’s venv activate docs, or use \`cmd\` + \`.venv\\Scripts\\activate.bat\`.
- \`pip\` install fails on network → retry; on mainland networks you may need a PyPI mirror.
- Torch is huge → wait; a CPU wheel is enough.

## Don't worry about yet

CUDA drivers, conda, Docker, or Hugging Face login (that comes when a chapter needs gated weights).

## Next

**Terminal survival** (\`?c=start-terminal\`).
`,
  },
  {
    id: 'start-terminal',
    ep: '00d',
    title: 'Terminal survival',
    summary: 'cd, list files, read errors, and run your first chapter script.',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 15,
    markdown: `## In one sentence

The terminal is a text window where you type commands; you only need a few of them to learn arche.

## Before you start

- Finish setup so \`.venv\` activates and \`ep00_check_setup.py\` prints OK.
- Time: ~15 minutes.

## The idea (plain language)

Useful commands:

| Goal | macOS / Linux | Windows (PowerShell) |
| --- | --- | --- |
| Where am I? | \`pwd\` | \`pwd\` |
| List files | \`ls\` | \`dir\` or \`ls\` |
| Enter folder | \`cd arche\` | \`cd arche\` |
| Go up one folder | \`cd ..\` | \`cd ..\` |

A **traceback** is Python’s way of saying “I crashed; here is the stack of calls.” Read the **last** paragraph first—it usually names the real error.

## In the code

First real lab sheet after setup:

\`\`\`bash
python chapters/01-inference/ep01_overview.py
\`\`\`

(You can wait until Part 1 if you prefer; running it early is encouraging.)

## Do this now

1. \`cd\` into the \`arche\` folder.
2. Activate \`.venv\`.
3. Run \`python chapters/00-start/ep00_check_setup.py\` again.
4. Optional: run \`python chapters/01-inference/ep01_overview.py\` and notice it prints config numbers—not a chatbot.

## If something breaks

- \`No module named torch\` → venv not active, or requirements not installed.
- \`can't open file\` → you are in the wrong folder; \`ls\`/\`dir\` should show \`chapters\` and \`arche\`.
- Copy the **last** error lines when asking for help.

## Don't worry about yet

Shell scripting, git branching, or editors beyond Notepad / TextEdit / VS Code basics.

## Next

**How to read a chapter script** (\`?c=start-reading-code\`).
`,
  },
  {
    id: 'start-reading-code',
    ep: '00e',
    title: 'How to read a chapter script',
    summary: 'Recognize imports, main, and prints—without writing Python yet.',
    script: 'chapters/00-start/ep00_check_setup.py',
    partId: 'start',
    estimatedMinutes: 12,
    markdown: `## In one sentence

You can follow arche by **recognizing patterns** in scripts long before you can write them.

## Before you start

- Terminal survival done.
- Time: ~12 minutes.
- Open \`chapters/00-start/ep00_check_setup.py\` in any text editor.

## The idea (plain language)

Look for these landmarks:

1. A triple-quoted blurb at the top — human instructions (how to run).
2. \`import ...\` — “borrow tools.”
3. \`def main():\` — “the steps live here.”
4. \`print(...)\` — “show me something.”
5. \`if __name__ == "__main__":\` — “only run main when I execute this file.”

You are not expected to invent new programs in Part 0. Reading + running is enough.

## In the code

In \`ep00_check_setup.py\`, \`main\` prints your Python version and whether \`torch\` imports. That is the whole job.

## Do this now

1. Open the file.
2. Point to \`def main\` with your cursor.
3. Point to the final \`OK\` print.
4. Close the file—you already “read” it successfully.

## If something breaks

- Editor shows weird symbols → open as UTF-8 text, not as a Word document.
- Fear of breaking code → these scripts are small; \`git checkout -- file\` restores them if you use git.

## Don't worry about yet

Classes, type hints, or PyTorch internals. Site chapters explain ideas first.

## Next

**Map of the journey** (\`?c=start-map\`).
`,
  },
  {
    id: 'start-map',
    ep: '00f',
    title: 'Map of the journey',
    summary: 'Parts 1–4 at a glance, Llama 3.2 lineage, and when Hugging Face login matters.',
    script: '',
    partId: 'start',
    estimatedMinutes: 10,
    markdown: `## In one sentence

You now know the map: trunk → faster inference → tiny training → frontier primers.

## Before you start

- Finish the other Part 0 chapters.
- Time: ~10 minutes.

## The idea (plain language)

**Llama 3.2 lineage** means: we copy the *shape* of a modern dense decoder (RMSNorm, RoPE, SwiGLU, GQA) used by Llama 3.2, then optionally peek at MoE / MLA ideas used in frontier systems.

\`\`\`mermaid
flowchart TB
  p0[Part 0 Start] --> p1[Part 1 Dense trunk]
  p1 --> p2[Part 2 Faster inference]
  p2 --> p3[Part 3 Tiny training]
  p3 --> p4[Part 4 Frontier]
\`\`\`

**Hugging Face login** is only needed when a chapter loads official tokenizer/weights (gated model). Many chapters run on random tiny tensors with no download.

## In the code

Library code lives under \`arche/\`. Chapters call into it. The site narrates.

## Do this now

1. Mark Part 0 chapters as read.
2. Press **Next** into Part 1 — Overview.

## If something breaks

- Feeling overwhelmed by Part 4 names (MoE, MLA) → ignore them until you arrive; Part 1 never requires them.

## Don't worry about yet

Reproducing industry training runs. This course builds understanding and small runnable pieces.

## Next

Part 1 begins: **Overview** (\`?c=overview\`).
`,
  },
];
