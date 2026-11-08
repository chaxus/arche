import 'ranui/button';
import 'ranui/card';
import 'ranui/markdown';
import 'ranui/math';
import 'ranui/select';
import 'ranui/theme-switch';
import { chaptersFor, type Chapter } from './content';
import type { PartId } from './content/types';
import {
  LOCALES,
  localeFromUrlPath,
  localeHref,
  prefixOf,
  type LocaleDef,
} from './i18n/locales';
import { uiFor } from './i18n/ui';

const PROGRESS_KEY = 'arche.chapter.done';

const root = document.querySelector('#app');
if (!root) {
  throw new Error('#app missing');
}
const app = root;

let locale = localeFromUrlPath(location.pathname);
let ui = uiFor(locale);
let chapters = chaptersFor(locale);

const params = new URLSearchParams(location.search);
let activeChapterId = params.get('c') ?? chapters[0]?.id ?? 'overview';

function homeHref(target: LocaleDef = locale): string {
  return localeHref('/', target);
}

function chapterUrl(id: string, target: LocaleDef = locale): string {
  const url = new URL(homeHref(target), location.origin);
  url.searchParams.set('c', id);
  return `${url.pathname}${url.search}`;
}

function themeSwitchAttrs(): string {
  if (locale.id === 'cn') {
    return 'label="主题" label-system="系统" label-light="浅色" label-dark="深色"';
  }
  return 'label="Theme" label-system="System" label-light="Light" label-dark="Dark"';
}

function readDone(): Set<string> {
  try {
    const raw = localStorage.getItem(PROGRESS_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw) as unknown;
    return new Set(Array.isArray(parsed) ? parsed.map(String) : []);
  } catch {
    return new Set();
  }
}

function writeDone(done: Set<string>): void {
  localStorage.setItem(PROGRESS_KEY, JSON.stringify([...done]));
}

function markDone(id: string): void {
  const done = readDone();
  done.add(id);
  writeDone(done);
}

function progressFraction(): { done: number; total: number } {
  const done = readDone();
  const total = chapters.length;
  const count = chapters.filter((chapter) => done.has(chapter.id)).length;
  return { done: count, total };
}

function wireLocaleSelect(): void {
  const select = document.querySelector('#locale-select') as
    | (HTMLElement & { value: string })
    | null;
  if (!select) return;

  select.addEventListener('change', (event) => {
    const detail = (event as CustomEvent<{ value?: string }>).detail;
    const nextId = detail?.value ?? select.value;
    const next = LOCALES.find((item) => item.id === nextId);
    if (!next || next.id === locale.id) return;
    location.assign(chapterUrl(activeChapterId, next));
  });
}

function renderShell(): void {
  const localeOptions = LOCALES.map(
    (item) => `<r-option value="${item.id}" lang="${item.lang}">${item.label}</r-option>`,
  ).join('');

  app.innerHTML = `
    <div class="shell">
      <header class="topbar">
        <div class="brand">
          <a href="${homeHref()}">arche</a>
          <span>${ui.brandTagline}</span>
        </div>
        <div class="topbar-actions">
          <r-select
            id="locale-select"
            class="locale-select"
            value="${locale.id}"
            trigger="click"
            aria-label="${ui.langMenu}"
          >
            ${localeOptions}
          </r-select>
          <r-theme-switch ${themeSwitchAttrs()}></r-theme-switch>
          <r-button type="primary" id="repo-link">${ui.github}</r-button>
        </div>
      </header>
      <div class="layout">
        <nav class="nav" aria-label="${ui.chaptersNav}" id="nav"></nav>
        <main class="article" id="article"></main>
      </div>
    </div>
  `;

  document.querySelector('#repo-link')?.addEventListener('click', () => {
    window.open('https://github.com/chaxus/arche', '_blank', 'noopener,noreferrer');
  });
  wireLocaleSelect();
}

function renderNav(activeId: string): void {
  const nav = document.querySelector('#nav') as HTMLElement;
  nav.innerHTML = '';
  const { done, total } = progressFraction();
  const pct = total === 0 ? 0 : Math.round((done / total) * 100);
  const progress = document.createElement('div');
  progress.className = 'progress';
  progress.innerHTML = `
    <div class="progress-label">
      <span>${ui.progressLabel}</span>
      <span>${done}/${total}</span>
    </div>
    <div class="progress-track" aria-hidden="true">
      <div class="progress-fill" style="width:${pct}%"></div>
    </div>
  `;
  nav.append(progress);

  const doneIds = readDone();
  let lastPart: PartId | null = null;
  for (const chapter of chapters) {
    if (chapter.partId !== lastPart) {
      lastPart = chapter.partId;
      const header = document.createElement('div');
      header.className = 'nav-part';
      header.textContent = ui.partLabels[chapter.partId];
      nav.append(header);
    }
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'nav-link';
    const tick = doneIds.has(chapter.id) ? '<span class="nav-done" aria-hidden="true">✓</span>' : '';
    btn.innerHTML = `${tick}<span class="nav-ep">${chapter.ep}</span>${chapter.title}`;
    if (chapter.id === activeId) {
      btn.setAttribute('aria-current', 'page');
    }
    btn.addEventListener('click', () => selectChapter(chapter.id));
    nav.append(btn);
  }
}

function fallbackChatTemplate(messages: { role: string; content: string }[]): string {
  const parts = messages.map((msg) => `${msg.role}: ${msg.content}`);
  parts.push('assistant:');
  return parts.join('\n');
}

function runChatDemo(host: HTMLElement): void {
  const out = host.querySelector('.demo-out') as HTMLElement;
  out.textContent = fallbackChatTemplate([
    { role: 'system', content: 'You teach modern dense LLMs.' },
    { role: 'user', content: 'Why RoPE instead of absolute PE?' },
  ]);
}

function runBatchDemo(host: HTMLElement): void {
  const out = host.querySelector('.demo-out') as HTMLElement;
  type Job = { id: string; tokens: number[]; maxNew: number; generated: number };
  const jobs: Job[] = [
    { id: 'a', tokens: [1, 2, 3], maxNew: 2, generated: 0 },
    { id: 'b', tokens: [7, 8], maxNew: 4, generated: 0 },
    { id: 'c', tokens: [9], maxNew: 1, generated: 0 },
  ];
  const lines: string[] = [];
  let step = 0;
  while (jobs.some((job) => job.generated < job.maxNew)) {
    step += 1;
    const active = jobs.filter((job) => job.generated < job.maxNew);
    for (const job of active) {
      job.tokens.push(job.tokens[job.tokens.length - 1] ?? 0);
      job.generated += 1;
    }
    lines.push(
      `step ${step}: active=${active.map((j) => j.id).join(',')} → ${jobs
        .map((j) => `${j.id}:[${j.tokens.join(',')}]`)
        .join(' ')}`,
    );
  }
  out.textContent = lines.join('\n');
}

function runTrainDemo(host: HTMLElement): void {
  const out = host.querySelector('.demo-out') as HTMLElement;
  let loss = 4.2;
  const rows = ['pretrain'];
  for (let i = 1; i <= 4; i += 1) {
    loss *= 0.82;
    rows.push(`  step ${i}: loss=${loss.toFixed(3)}`);
  }
  rows.push('SFT loss=1.84');
  rows.push('DPO loss=0.693');
  out.textContent = rows.join('\n');
}

function mountDemo(chapter: Chapter, host: HTMLElement): void {
  if (!chapter.demo) return;
  host.hidden = false;
  const hints: Record<NonNullable<Chapter['demo']>, string> = {
    chat: ui.demoChatHint,
    batch: ui.demoBatchHint,
    train: ui.demoTrainHint,
  };
  const labels: Record<NonNullable<Chapter['demo']>, string> = {
    chat: ui.demoChatRun,
    batch: ui.demoBatchRun,
    train: ui.demoTrainRun,
  };
  host.innerHTML = `
    <r-card heading="${ui.demoHeading}" description="ep${chapter.ep}">
      <p class="demo-hint">${hints[chapter.demo]}</p>
      <r-button type="primary" id="demo-run">${labels[chapter.demo]}</r-button>
      <pre class="demo-out" aria-live="polite"></pre>
    </r-card>
  `;
  const run = () => {
    if (chapter.demo === 'chat') runChatDemo(host);
    if (chapter.demo === 'batch') runBatchDemo(host);
    if (chapter.demo === 'train') runTrainDemo(host);
  };
  host.querySelector('#demo-run')?.addEventListener('click', run);
}

function neighborIds(chapterId: string): { prev?: Chapter; next?: Chapter } {
  const idx = chapters.findIndex((item) => item.id === chapterId);
  if (idx < 0) return {};
  return {
    prev: idx > 0 ? chapters[idx - 1] : undefined,
    next: idx < chapters.length - 1 ? chapters[idx + 1] : undefined,
  };
}

function renderArticle(chapter: Chapter): void {
  const article = document.querySelector('#article') as HTMLElement;
  const done = readDone().has(chapter.id);
  const { prev, next } = neighborIds(chapter.id);
  const scriptLine = chapter.script
    ? `${ui.companionPython}: <code>python ${chapter.script}</code> · `
    : '';
  article.innerHTML = `
    <div class="article-head">
      <div>
        <p class="part-eyebrow">${ui.partLabels[chapter.partId]} · ${chapter.estimatedMinutes} ${ui.minutesLabel}</p>
        <h1>${chapter.title}</h1>
      </div>
      <r-button id="mark-done" type="${done ? 'text' : 'primary'}">${
        done ? ui.markedDone : ui.markDone
      }</r-button>
    </div>
    <p class="lede">${chapter.summary}</p>
    <r-card heading="${ui.lessonHeading}" description="ep${chapter.ep}">
      <div class="panel" id="md-host"></div>
    </r-card>
    <div class="demo-wrap" id="demo-host" hidden></div>
    <div class="math-wrap" id="math-host" hidden></div>
    <div class="chapter-nav">
      <r-button id="prev-chapter" type="text" ${prev ? '' : 'disabled'}>${ui.prevChapter}${
        prev ? `: ${prev.title}` : ''
      }</r-button>
      <r-button id="next-chapter" type="primary" ${next ? '' : 'disabled'}>${ui.nextChapter}${
        next ? `: ${next.title}` : ''
      }</r-button>
    </div>
    <p class="footer-note">
      ${scriptLine}${ui.siteLabel}:
      <a href="https://arche.chaxus.com${prefixOf(locale) || '/'}">arche.chaxus.com${prefixOf(locale)}</a>
    </p>
  `;

  const md = document.createElement('r-markdown');
  md.setAttribute('highlight', '');
  (md as HTMLElement & { content: string }).content = chapter.markdown;
  document.querySelector('#md-host')?.append(md);

  const mathHost = document.querySelector('#math-host') as HTMLElement | null;
  if (mathHost && chapter.latex) {
    mathHost.hidden = false;
    const math = document.createElement('r-math') as HTMLElement & { latex: string };
    math.latex = chapter.latex;
    mathHost.append(math);
  }

  const demoHost = document.querySelector('#demo-host') as HTMLElement | null;
  if (demoHost) mountDemo(chapter, demoHost);

  document.querySelector('#mark-done')?.addEventListener('click', () => {
    markDone(chapter.id);
    renderNav(chapter.id);
    renderArticle(chapter);
  });
  document.querySelector('#prev-chapter')?.addEventListener('click', () => {
    if (prev) selectChapter(prev.id);
  });
  document.querySelector('#next-chapter')?.addEventListener('click', () => {
    if (next) selectChapter(next.id);
  });
}

function applyDocumentLang(): void {
  document.documentElement.lang = locale.lang;
  document.title =
    locale.id === 'cn'
      ? 'arche — 从第一性原理手写现代 dense'
      : 'arche — modern dense from first principles';
}

function selectChapter(id: string): void {
  const chapter = chapters.find((item) => item.id === id) ?? chapters[0];
  if (!chapter) return;
  activeChapterId = chapter.id;
  const url = new URL(chapterUrl(chapter.id), location.origin);
  history.replaceState(null, '', `${url.pathname}${url.search}`);
  params.set('c', chapter.id);
  renderNav(chapter.id);
  renderArticle(chapter);
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
}

applyDocumentLang();
renderShell();
selectChapter(activeChapterId);
