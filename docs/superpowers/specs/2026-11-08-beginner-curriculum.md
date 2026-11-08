# Spec: Beginner-first learning site (absolute beginners)

**Status:** implemented  


**Date:** 2026-11-08  
**Audience decision:** Type 2 — little or no Python, no ML/LLM background  

## Problem

The arche learning site ([arche.chaxus.com](https://arche.chaxus.com)) currently shows short technical summaries per chapter. That works as a syllabus for people who already know Transformers. Absolute beginners bounce: undefined terms, no environment setup, no “what am I looking at,” and 23 flat nav items with no on-ramp.

## Goal

A complete beginner can open the site, set up a machine with copy-paste steps, understand what an LLM is doing in plain language, then follow chapters in order to the modern dense trunk (and optional frontier chapters) without needing another course first.

**Success looks like:**

- Part 0 alone is enough to install Python, create a venv, run `ep01`, and read the terminal output without panic.
- Every chapter answers: what problem, plain explanation, what to type, what you should see, what to ignore for now, what comes next.
- Chinese and English stay in sync and equally beginner-friendly.
- Experts can still skim; depth is additive, not a second site.

## Non-goals

- Not a general Python textbook (only the minimum needed to run this repo).
- Not reinventing BPE / CUDA kernels / full vLLM.
- Not video or interactive Jupyter in v1 (keep ranui + companion `.py` scripts).
- Not rewriting the PyTorch library API for pedagogy (site + chapter scripts carry the teaching load).

## Audience assumptions

| Can assume | Must not assume |
| --- | --- |
| Can install apps, open a terminal with guidance | Knows Python syntax |
| Willing to copy-paste commands | Knows tensors, gradients, attention |
| Has (or can get) a laptop with internet | Has a GPU |
| Reads EN or 中文 on the site | Has used Hugging Face before |

## Information architecture

### Nav parts (ordered)

1. **Part 0 — Start here** (new; required first)  
2. **Part 1 — The dense trunk** (existing ep01–11, rewritten)  
3. **Part 2 — Faster inference** (ep12–17)  
4. **Part 3 — Tiny training** (ep18–19)  
5. **Part 4 — Frontier primers** (ep20–23)

Nav UI shows part headers + chapter links. Progress meter stays global.

### Part 0 chapters (new)

| id | Title (EN) | Purpose |
| --- | --- | --- |
| `start-welcome` | Welcome | What arche is; what you will build; time expectations; how the site works (progress, language switch) |
| `start-llm-idea` | What an LLM does | Next-token prediction in plain language; characters vs tokens vs vectors (no formulas yet) |
| `start-setup` | Set up your computer | Install Python 3.10+, clone repo, venv, `pip install -r requirements.txt`, common errors |
| `start-terminal` | Terminal survival | `cd`, `ls`/`dir`, what a traceback is, how to copy errors; run `ep01` successfully |
| `start-reading-code` | How to read a chapter script | File path, `import`, `def main`, `print`; you do not need to write Python yet—only recognize patterns |
| `start-map` | Map of the journey | One diagram of Parts 1–4; what “Llama 3.2 lineage” means in one paragraph; when you need Hugging Face login |

Part 0 lives primarily on the **site** (markdown). Optional: one `chapters/00-start/ep00_check_setup.py` that prints Python version / torch import / repo path for confidence.

### Chapter body template (Parts 1–4)

Every rewritten chapter markdown MUST include these sections in order (localized):

1. **In one sentence** — what this chapter unlocks  
2. **Before you start** — prerequisites (link earlier chapters); time estimate  
3. **The idea (plain language)** — analogy first, then precise terms  
4. **Picture / formula (optional)** — mermaid and/or one latex block only if it helps  
5. **In the code** — which file under `arche/`, what the companion script prints  
6. **Do this now** — exact commands; expected output shape (not necessarily full dump)  
7. **If something breaks** — 1–3 common failures  
8. **Don’t worry about yet** — park advanced digressions  
9. **Next** — explicit link to the next chapter id  

Summaries in the nav stay short; the article carries the depth.

## Content depth guidelines

- Prefer short paragraphs and numbered steps over dense walls of text.
- Define a jargon word the first time it appears in that language.
- Use the same metaphor family across early chapters where possible (e.g. “dictionary of vectors,” “spotlight over previous words”).
- Code citations: name the module; quote at most a few lines in fenced blocks when essential.
- Hugging Face gated steps appear only when required (tokenizer / weights / align); Part 0 explains login once; later chapters link back.

## Site / code changes in scope

| Area | Change |
| --- | --- |
| `web/src/content/types.ts` | Add `part: string` (or `partId`) on chapters; optional `estimatedMinutes` |
| `web/src/content/en.ts` / `cn.ts` | Part 0 chapters + full rewrite of ep01–23 bodies to the template |
| `web/src/i18n/ui.ts` | Part labels, “Start here”, template section headings if shared in UI |
| `web/src/main.ts` + `styles.css` | Render part group headers in nav; optional prev/next chapter buttons |
| `chapters/00-start/` | Optional `ep00_check_setup.py` |
| Companion scripts ep01+ | Lighten intimidation: clearer prints / comments at top; no behavior change required |
| `README.md` | Point absolute beginners to the site Part 0 first |

## Out of scope for this spec

- Auto-grading quizzes  
- Account / cloud notebook hosting  
- Translating into languages beyond EN/CN  
- Changing the arche PyTorch model math  

## Rollout

1. Ship Part 0 + nav parts + template applied to ep01–03 (prove the shape).  
2. Rewrite remaining Part 1 (ep04–11).  
3. Parts 2–4.  
4. Pass of companion script print polish + README.  
5. Deploy Pages.

Commits stay one-feature / one coherent chunk where practical (e.g. “Add Part 0”, “Rewrite trunk chapters 01–03”, …).

## Risks

| Risk | Mitigation |
| --- | --- |
| Essays too long → bounce | Template sections; keep “Don’t worry about yet”; scannable headings |
| EN/CN drift | Rewrite in pairs per chapter in the same change set |
| Setup docs rot across OS | Cover macOS + Windows Terminal + generic Linux; link OS-specific only where needed |
| Scope explosion into full ML course | Hard non-goals; Part 0 is “enough to run arche,” not CS101 |

## Approval

Approve this spec to unlock an implementation plan under `docs/superpowers/plans/`.  
Reply with **可以** / requested edits.
