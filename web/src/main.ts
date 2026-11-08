import 'ranui/button';
import 'ranui/card';
import 'ranui/markdown';
import 'ranui/math';
import 'ranui/select';
import 'ranui/theme-switch';
import { chaptersFor, type Chapter } from './content';
import { chaptersInPartOrder, type PartId } from './content/types';
import {
  mountEmbeddingPlayground,
  mountPretrainPlayground,
  mountTokenizerPlayground,
} from './features/playgrounds';
import { mountVisualizer, type VisualizerKind } from './features/visualizers';
import { mountSource } from './features/source';
import { LOCALES, type LocaleDef } from './i18n/locales';
import { uiFor } from './i18n/ui';
import {
  HOME_ID,
  ORIGIN,
  SITE,
  pagePath,
  pageUrl,
  routeFromPathname,
} from './site';

const PROGRESS_KEY = 'arche.chapter.done';

const root = document.querySelector('#app');
if (!root) {
  throw new Error('#app missing');
}
const app = root;

function resolveBoot(): { locale: LocaleDef; chapterId: string } {
  const params = new URLSearchParams(location.search);
  const legacy = params.get('c');
  const localeOnly = routeFromPathname(location.pathname, new Set());
  const ids = new Set(
    chaptersInPartOrder(chaptersFor(localeOnly.locale)).map((chapter) => chapter.id),
  );
  const route = routeFromPathname(location.pathname, ids);

  if (legacy && legacy !== HOME_ID && ids.has(legacy)) {
    history.replaceState(null, '', pagePath(legacy, route.locale));
    return { locale: route.locale, chapterId: legacy };
  }
  if (legacy) {
    history.replaceState(null, '', pagePath(route.chapterId, route.locale));
  }
  return { locale: route.locale, chapterId: route.chapterId };
}

const boot = resolveBoot();
let locale = boot.locale;
let ui = uiFor(locale);
let chapters = chaptersInPartOrder(chaptersFor(locale));
let activeChapterId = boot.chapterId;
let showingHome = activeChapterId === HOME_ID;

function chapterUrl(id: string, target: LocaleDef = locale): string {
  return pagePath(id, target);
}

function syncHead(chapter: Chapter | null): void {
  const title = chapter
    ? `${chapter.title} — arche`
    : locale.id === 'cn'
      ? 'arche — 从零开始，亲手写出大模型'
      : 'arche — build an LLM by hand, from zero';
  const description = chapter
    ? chapter.summary
    : locale.id === 'cn'
      ? SITE.descriptionCn
      : SITE.descriptionEn;
  document.title = title;

  const setMeta = (selector: string, attr: string, value: string) => {
    const el = document.querySelector(selector);
    if (el) el.setAttribute(attr, value);
  };
  setMeta('meta[name="description"]', 'content', description);
  setMeta('meta[property="og:title"]', 'content', title);
  setMeta('meta[property="og:description"]', 'content', description);
  setMeta('meta[property="og:url"]', 'content', pageUrl(activeChapterId, locale));
  setMeta('link[rel="canonical"]', 'href', pageUrl(activeChapterId, locale));

  for (const other of LOCALES) {
    const hreflang = other.lang;
    const link = document.querySelector(`link[rel="alternate"][hreflang="${hreflang}"]`);
    if (link) link.setAttribute('href', pageUrl(activeChapterId, other));
  }
  const xDefault = document.querySelector('link[rel="alternate"][hreflang="x-default"]');
  if (xDefault) {
    const root = LOCALES.find((item) => item.dir === '') ?? locale;
    xDefault.setAttribute('href', pageUrl(activeChapterId, root));
  }
  void ORIGIN;
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

function continueTarget(): Chapter {
  const done = readDone();
  const next = chapters.find((chapter) => !done.has(chapter.id));
  return next ?? chapters[0]!;
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
    const target = showingHome ? HOME_ID : activeChapterId;
    location.assign(chapterUrl(target, next));
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
          <a href="${chapterUrl(HOME_ID)}" id="brand-home">arche</a>
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
    window.open(SITE.github, '_blank', 'noopener,noreferrer');
  });
  document.querySelector('#brand-home')?.addEventListener('click', (event) => {
    event.preventDefault();
    selectHome();
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

  const homeLink = document.createElement('a');
  homeLink.href = chapterUrl(HOME_ID);
  homeLink.className = 'nav-link';
  homeLink.textContent = ui.homeNav;
  if (showingHome) homeLink.setAttribute('aria-current', 'page');
  homeLink.addEventListener('click', (event) => {
    event.preventDefault();
    selectHome();
  });
  nav.append(homeLink);

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
    const link = document.createElement('a');
    link.href = chapterUrl(chapter.id);
    link.className = 'nav-link';
    const tick = doneIds.has(chapter.id) ? '<span class="nav-done" aria-hidden="true">✓</span>' : '';
    link.innerHTML = `${tick}<span class="nav-ep">${chapter.ep}</span>${chapter.title}`;
    if (!showingHome && chapter.id === activeId) {
      link.setAttribute('aria-current', 'page');
    }
    link.addEventListener('click', (event) => {
      event.preventDefault();
      selectChapter(chapter.id);
    });
    nav.append(link);
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
  const loc = locale.id === 'cn' ? 'cn' : 'en';
  if (chapter.demo === 'tokenizer') {
    mountTokenizerPlayground(host, loc);
    return;
  }
  if (chapter.demo === 'embedding') {
    mountEmbeddingPlayground(host, loc);
    return;
  }
  if (chapter.demo === 'pretrain') {
    mountPretrainPlayground(host, loc);
    return;
  }
  const vizKinds: VisualizerKind[] = ['rope', 'gqa', 'kv'];
  if (vizKinds.includes(chapter.demo as VisualizerKind)) {
    const card = document.createElement('div');
    card.innerHTML = `<r-card heading="${ui.demoVizHeading}" description="ep${chapter.ep}"></r-card>`;
    const inner = document.createElement('div');
    inner.className = 'viz-card-body';
    card.querySelector('r-card')?.append(inner);
    host.hidden = false;
    host.replaceChildren(card);
    mountVisualizer(inner, chapter.demo as VisualizerKind, locale.id);
    return;
  }

  host.hidden = false;
  const hints: Record<string, string> = {
    chat: ui.demoChatHint,
    batch: ui.demoBatchHint,
    train: ui.demoTrainHint,
  };
  const labels: Record<string, string> = {
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

function renderHome(): void {
  const article = document.querySelector('#article') as HTMLElement;
  const target = continueTarget();
  const done = readDone();
  const { done: doneCount, total } = progressFraction();
  const hasProgress = doneCount > 0;
  const continueText = hasProgress
    ? `${ui.continueLabel}: ${target.title}`
    : ui.continueStart;

  const partBlocks: string[] = [];
  let current: PartId | null = null;
  let buffer: Chapter[] = [];
  const flush = () => {
    if (!current || buffer.length === 0) return;
    const read = buffer.filter((c) => done.has(c.id)).length;
    const mins = buffer.reduce((sum, c) => sum + c.estimatedMinutes, 0);
    partBlocks.push(`
      <div class="home-part">
        <div class="home-part-head">
          <strong>${ui.partLabels[current]}</strong>
          <span>${read}/${buffer.length} ${ui.homeDoneLabel} · ~${mins} ${ui.minutesLabel}</span>
        </div>
        <a class="home-part-go" href="${chapterUrl(buffer[0]!.id)}" data-id="${buffer[0]!.id}">${buffer[0]!.title}</a>
      </div>
    `);
  };
  for (const chapter of chapters) {
    if (chapter.partId !== current) {
      flush();
      current = chapter.partId;
      buffer = [chapter];
    } else {
      buffer.push(chapter);
    }
  }
  flush();

  article.innerHTML = `
    <p class="part-eyebrow">${ui.homeNav}</p>
    <h1>${ui.homeTitle}</h1>
    <p class="lede">${ui.homeLede}</p>
    <div class="home-continue">
      <r-button type="primary" id="continue-btn">${continueText}</r-button>
      <span class="home-progress-meta">${doneCount}/${total}</span>
    </div>
    <h2 class="home-parts-title">${ui.homePartsHeading}</h2>
    <div class="home-parts">${partBlocks.join('')}</div>
  `;

  document.querySelector('#continue-btn')?.addEventListener('click', () => {
    selectChapter(target.id);
  });
  article.querySelectorAll('.home-part-go').forEach((node) => {
    node.addEventListener('click', (event) => {
      event.preventDefault();
      const id = (node as HTMLElement).dataset.id;
      if (id) selectChapter(id);
    });
  });
}

function splitLessonMarkdown(md: string): { body: string; fix: string } {
  const heading = md.match(/^##\s+(Fix|翻车)\s*$/m);
  if (!heading || heading.index == null) {
    const inline = md.match(/\n\*\*(?:Fix|翻车)[：:]\*\*\s*([^\n]+)/);
    if (inline && inline.index != null) {
      return {
        body: `${md.slice(0, inline.index)}${md.slice(inline.index + inline[0].length)}`.trimEnd() + '\n',
        fix: (inline[1] ?? '').trim(),
      };
    }
    return { body: md, fix: '' };
  }
  const after = md.slice(heading.index + heading[0].length);
  const next = after.match(/\n##\s+/);
  if (next && next.index != null) {
    return {
      body: `${md.slice(0, heading.index)}${after.slice(next.index + 1)}`.trimEnd() + '\n',
      fix: after.slice(0, next.index).trim(),
    };
  }
  return {
    body: md.slice(0, heading.index).trimEnd() + '\n',
    fix: after.trim(),
  };
}

function renderArticle(chapter: Chapter): void {
  const article = document.querySelector('#article') as HTMLElement;
  const done = readDone().has(chapter.id);
  const { prev, next } = neighborIds(chapter.id);
  const { body: lessonBody, fix: fixMd } = splitLessonMarkdown(chapter.markdown);
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
    <div class="src-wrap" id="src-host" hidden></div>
    <div class="demo-wrap" id="demo-host" hidden></div>
    <div class="math-wrap" id="math-host" hidden></div>
    ${
      fixMd
        ? `<details class="fix-box"><summary>${ui.fixHeading}</summary><div class="panel" id="fix-host"></div></details>`
        : ''
    }
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
      <a href="${pageUrl(chapter.id, locale)}">${ORIGIN.replace('https://', '')}${pagePath(chapter.id, locale)}</a>
    </p>
  `;

  const md = document.createElement('r-markdown');
  md.setAttribute('highlight', '');
  (md as HTMLElement & { content: string }).content = lessonBody;
  document.querySelector('#md-host')?.append(md);

  if (fixMd) {
    const fixEl = document.createElement('r-markdown');
    fixEl.setAttribute('highlight', '');
    (fixEl as HTMLElement & { content: string }).content = fixMd;
    document.querySelector('#fix-host')?.append(fixEl);
  }

  const srcHost = document.querySelector('#src-host') as HTMLElement | null;
  if (srcHost) {
    mountSource(srcHost, chapter.id, chapter.script, locale.id, {
      heading: ui.sourceHeading,
      libraryTab: ui.sourceLibraryTab,
      labTab: ui.sourceLabTab,
      empty: ui.sourceEmpty,
      github: ui.sourceGithub,
    });
  }

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
}

function selectHome(push = true): void {
  showingHome = true;
  activeChapterId = HOME_ID;
  if (push) history.pushState(null, '', chapterUrl(HOME_ID));
  else history.replaceState(null, '', chapterUrl(HOME_ID));
  renderNav(HOME_ID);
  renderHome();
  syncHead(null);
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
}

function selectChapter(id: string, push = true): void {
  if (id === HOME_ID) {
    selectHome(push);
    return;
  }
  const chapter = chapters.find((item) => item.id === id) ?? chapters[0];
  if (!chapter) return;
  showingHome = false;
  activeChapterId = chapter.id;
  if (push) history.pushState(null, '', chapterUrl(chapter.id));
  else history.replaceState(null, '', chapterUrl(chapter.id));
  renderNav(chapter.id);
  renderArticle(chapter);
  syncHead(chapter);
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
}

function onPopState(): void {
  const ids = new Set(chapters.map((chapter) => chapter.id));
  const route = routeFromPathname(location.pathname, ids);
  if (route.locale.id !== locale.id) {
    location.assign(location.pathname);
    return;
  }
  if (route.chapterId === HOME_ID) selectHome(false);
  else selectChapter(route.chapterId, false);
}

applyDocumentLang();
renderShell();
window.addEventListener('popstate', onPopState);
if (showingHome || !chapters.some((c) => c.id === activeChapterId)) {
  selectHome(false);
} else {
  selectChapter(activeChapterId, false);
}
