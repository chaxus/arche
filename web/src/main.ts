import 'ranui/button';
import 'ranui/card';
import 'ranui/markdown';
import 'ranui/math';
import 'ranui/select';
import 'ranui/theme-switch';
import { chaptersFor, type Chapter } from './content';
import {
  LOCALES,
  localeFromUrlPath,
  localeHref,
  prefixOf,
  type LocaleDef,
} from './i18n/locales';
import { uiFor } from './i18n/ui';

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
  for (const chapter of chapters) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'nav-link';
    btn.innerHTML = `<span class="nav-ep">${chapter.ep}</span>${chapter.title}`;
    if (chapter.id === activeId) {
      btn.setAttribute('aria-current', 'page');
    }
    btn.addEventListener('click', () => selectChapter(chapter.id));
    nav.append(btn);
  }
}

function renderArticle(chapter: Chapter): void {
  const article = document.querySelector('#article') as HTMLElement;
  article.innerHTML = `
    <h1>${chapter.title}</h1>
    <p class="lede">${chapter.summary}</p>
    <r-card heading="${ui.lessonHeading}" description="ep${chapter.ep}">
      <div class="panel" id="md-host"></div>
    </r-card>
    <div class="math-wrap" id="math-host" hidden></div>
    <p class="footer-note">
      ${ui.companionPython}:
      <code>python chapters/01-inference/ep${chapter.ep}_*.py</code>
      · ${ui.siteLabel}:
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
