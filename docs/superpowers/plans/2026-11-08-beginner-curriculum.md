# Beginner-first learning site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the arche learning site into an absolute-beginner path (no Python assumed): Part 0 onboarding, part-grouped nav, and every chapter rewritten to a fixed nine-section template in EN and CN.

**Architecture:** Content stays in `web/src/content/{en,cn}.ts` with a `partId` on each chapter; `main.ts` groups the nav by part and adds prev/next. A tiny Node content lint script asserts every chapter has the required markdown headings. Companion Python only gains an `ep00_check_setup.py` plus light header comments—no model math changes.

**Tech Stack:** Vite + TypeScript + ranui, existing chapter scripts, pytest unchanged for model code.

**Spec:** `docs/superpowers/specs/2026-11-08-beginner-curriculum.md`

## Global Constraints

- Audience type 2: no Python / no ML assumed; Part 0 teaches only enough to run this repo.
- Chapter markdown section order (EN headings; CN must use the paired localized headings listed in Task 1): In one sentence → Before you start → The idea (plain language) → Picture / formula (optional) → In the code → Do this now → If something breaks → Don't worry about yet → Next.
- EN and CN rewritten in the same commit for any chapter batch.
- Do not change arche model numerics or public Python APIs except additive `chapters/00-start/ep00_check_setup.py`.
- Keep demos / progress / language switcher behavior.
- One coherent commit per task below.

## Review Focus

- Missing required heading in one locale → content lint fails in that task.
- Nav part order wrong or Part 0 not first → UI smoke via `npm run build` + manual check of first nav label.
- Chapter `script` missing for Part 0 optional script-only chapter → allow empty/`ep00` only where specified.
- Prev/next skips Part boundaries incorrectly → unit-free check: next of last Part 0 chapter is first Part 1 id `overview`.
- Windows vs macOS setup copy → Part 0 setup chapter must mention both Terminal and Windows Terminal in EN and CN.

---

### Task 1: Chapter schema, part labels, nav groups, prev/next

**Files:**
- Modify: `web/src/content/types.ts`
- Modify: `web/src/i18n/ui.ts`
- Modify: `web/src/main.ts`
- Modify: `web/src/styles.css`
- Modify: `web/src/content/en.ts` (add `partId` + `estimatedMinutes` to every existing chapter; bodies unchanged yet)
- Modify: `web/src/content/cn.ts` (same)
- Create: `web/scripts/lint-content.mjs`
- Modify: `web/package.json` (add `"lint:content": "node scripts/lint-content.mjs"`)

**Interfaces:**
- Consumes: existing `Chapter` fields (`id`, `ep`, `title`, `summary`, `markdown`, `script`, …)
- Produces:
  - `partId: 'start' | 'trunk' | 'infer' | 'train' | 'frontier'`
  - `estimatedMinutes: number`
  - `UiCopy.partLabels: Record<partId, string>`
  - `UiCopy.prevChapter` / `nextChapter` strings
  - CN required H2 set: `一句话` / `开始之前` / `用人话讲` / `图或公式` (optional) / `对应代码` / `现在就做` / `如果翻车了` / `先别深挖` / `下一章`
  - EN required H2 set: exact English strings from Global Constraints (Picture / formula optional)
  - `lint-content.mjs` exits 0 only if every chapter in en+cn has `partId`, matching ids across locales, and all non-optional H2s present in `markdown`

- [ ] **Step 1: Extend `Chapter` and UI copy; map existing ep01–23 to partIds** (`start` unused until Task 2)

  Mapping: ep01–11 → `trunk`, ep12–17 → `infer`, ep18–19 → `train`, ep20–23 → `frontier`. Keep current markdown.

- [ ] **Step 2: Implement nav part headers + prev/next in `main.ts` / `styles.css`**

  When iterating chapters, insert a non-button header when `partId` changes. Article footer: buttons calling `selectChapter` for adjacent indices.

- [ ] **Step 3: Add `lint-content.mjs` that parses both content files for `id:` / `partId:` / `markdown:` blocks sufficiently to check H2s**

  For this task only, either skip H2 checks for chapters not yet rewritten **or** temporarily do not require H2s until Task 3—**prefer:** lint checks structural fields now; H2 requirements enabled via `--strict-template` flag that Tasks 3+ turn on in `package.json` as `"lint:content": "node scripts/lint-content.mjs --strict-template"` after first rewritten batch. Until Task 3, run without `--strict-template`.

- [ ] **Step 4: Verify**

  Run: `cd web && npm run build`  
  Expected: success.

  Run: `cd web && node scripts/lint-content.mjs`  
  Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add web/src/content/types.ts web/src/i18n/ui.ts web/src/main.ts web/src/styles.css web/src/content/en.ts web/src/content/cn.ts web/scripts/lint-content.mjs web/package.json
git commit -m "$(cat <<'EOF'
Add part-grouped nav and chapter schema for the beginner curriculum.

EOF
)"
```

---

### Task 2: Part 0 site chapters + setup check script

**Files:**
- Create: `chapters/00-start/ep00_check_setup.py`
- Modify: `web/src/content/en.ts` (prepend 6 Part 0 chapters)
- Modify: `web/src/content/cn.ts` (same ids/order)
- Test: run ep00 by hand

**Interfaces:**
- Consumes: Task 1 `partId: 'start'`
- Produces chapter ids in order: `start-welcome`, `start-llm-idea`, `start-setup`, `start-terminal`, `start-reading-code`, `start-map`
- `ep` labels: `00a`…`00f` (string) or `0.1`…`0.6` — use `00a`–`00f` for nav compactness
- `script`: empty string `''` for pure-site chapters; `start-terminal` and/or `start-setup` may point at `chapters/00-start/ep00_check_setup.py`
- `ep00_check_setup.py` prints: Python version, whether `torch` imports, absolute repo root, and a one-line “OK — continue to ep01”

- [ ] **Step 1: Write `ep00_check_setup.py` and run it**

  Run: `.venv/bin/python chapters/00-start/ep00_check_setup.py`  
  Expected: ends with OK line; no traceback.

- [ ] **Step 2: Author EN+CN Part 0 markdown (full template; setup covers macOS, Windows Terminal, Linux)**

  Default landing chapter becomes `start-welcome` (change `main.ts` fallback from `overview` to `start-welcome`).

- [ ] **Step 3: Verify**

  Run: `cd web && node scripts/lint-content.mjs && npm run build`  
  Expected: exit 0 / build ok.

- [ ] **Step 4: Commit**

```bash
git add chapters/00-start/ep00_check_setup.py web/src/content/en.ts web/src/content/cn.ts web/src/main.ts
git commit -m "$(cat <<'EOF'
Add Part 0 onboarding for absolute beginners.

EOF
)"
```

---

### Task 3: Rewrite trunk chapters ep01–03 (prove template) + enable strict lint

**Files:**
- Modify: `web/src/content/en.ts` (overview, tokenizer, embedding)
- Modify: `web/src/content/cn.ts` (same)
- Modify: `web/package.json` (`lint:content` adds `--strict-template`)
- Modify: `web/scripts/lint-content.mjs` if flag not finished in Task 1

**Interfaces:**
- Consumes: template H2 sets from Task 1
- Produces: three chapters fully templated; `Next` points to the next chapter id in prose (markdown link text ok; optional `?c=` mentioned)

- [ ] **Step 1: Rewrite ep01–03 EN+CN to the nine-section template (Picture optional)**

- [ ] **Step 2: Enable `--strict-template` in npm script; fix any Part 0 gaps the linter finds**

- [ ] **Step 3: Verify**

  Run: `cd web && npm run lint:content && npm run build`  
  Expected: Part 0 + ep01–03 pass strict; other chapters still fail **unless** strict mode only applies to chapters that already contain `## In one sentence` / `## 一句话` — **implement linter as:** if a chapter has any template H2, require the full required set; chapters with none are “legacy” until rewritten. After Task 3, Part 0 + 01–03 must be templated; later chapters remain legacy until their task.

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
Rewrite overview, tokenizer, and embedding for absolute beginners.

EOF
)"
```

---

### Task 4: Rewrite Part 1 remainder (ep04–11)

**Files:**
- Modify: `web/src/content/en.ts`, `web/src/content/cn.ts` for rmsnorm → generate (ids: `rmsnorm`, `rope`, `gqa`, `swiglu`, `block`, `stack`, `weights`, `generate`)

- [ ] **Step 1: Rewrite all eight chapters EN+CN to the template**

  HF login chapters (`tokenizer` already done; `weights` / `generate`) link back to Part 0 map / setup rather than re-teaching install.

- [ ] **Step 2: Verify** `cd web && npm run lint:content && npm run build`

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
Rewrite dense-trunk chapters RMSNorm through generation for beginners.

EOF
)"
```

---

### Task 5: Rewrite Part 2 (ep12–17)

**Files:**
- Modify: EN/CN for `kv-cache`, `rope-scaling`, `moe`, `mla`, `device-quant-chat`, `continuous-batching`

- [ ] **Step 1: Rewrite six chapters EN+CN**

- [ ] **Step 2: Verify** lint + build

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
Rewrite inference chapters KV cache through continuous batching for beginners.

EOF
)"
```

---

### Task 6: Rewrite Parts 3–4 (ep18–23)

**Files:**
- Modify: EN/CN for `pretrain`, `sft-dpo`, `frontier-ckpt`, `hf-align`, `sharded-frontier`, `paged-cuda-graph`

- [ ] **Step 1: Rewrite six chapters EN+CN**

- [ ] **Step 2: Verify** lint + build — at this point **all** chapters should be templated; optionally flip linter to require template on every chapter

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
Rewrite training and frontier chapters for absolute beginners.

EOF
)"
```

---

### Task 7: Companion script polish, README, deploy

**Files:**
- Modify: `README.md` (absolute beginners → site Part 0 first; list Part 0 / ep00)
- Modify: top-of-file comments / first prints in `chapters/01-inference/ep01_overview.py` and `chapters/00-start/ep00_check_setup.py` only unless a script’s first lines are hostile—then add a 3-line “If you are new…” comment to each `ep*.py` without changing logic
- Deploy: `cd web && npm run build && npx wrangler pages deploy dist --project-name=arche --branch=main`
- Push commits

- [ ] **Step 1: README + light script headers**

- [ ] **Step 2: Mark spec status approved/implemented in `docs/superpowers/specs/2026-11-08-beginner-curriculum.md`**

- [ ] **Step 3: Full verify**

  Run: `cd web && npm run lint:content && npm run build`  
  Run: `.venv/bin/pytest -q -m "not integration"`  
  Expected: content lint 0, build ok, pytest green.

- [ ] **Step 4: Commit, push, deploy**

```bash
git commit -m "$(cat <<'EOF'
Point beginners at Part 0 and polish chapter script entry comments.

EOF
)"
git push origin main
```

---

## Self-review notes

- Spec Part 0 table, template, nav parts, ep00, README, EN/CN sync, rollout order → Tasks 1–7.
- Strict lint phased so mid-rewrite tree stays shippable.
- No model-code tasks (per non-goals).
