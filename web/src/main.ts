import 'ranui/button';
import 'ranui/card';
import 'ranui/markdown';
import 'ranui/math';
import 'ranui/theme-switch';
import { CHAPTERS, type Chapter } from './content';

const app = document.querySelector('#app');
if (!app) {
  throw new Error('#app missing');
}

const params = new URLSearchParams(location.search);
const initialId = params.get('c') ?? CHAPTERS[0]?.id ?? 'overview';

app.innerHTML = `
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <a href="/">arche</a>
        <span>ἀρχή — modern dense from first principles</span>
      </div>
      <div class="topbar-actions">
        <r-theme-switch></r-theme-switch>
        <r-button type="primary" id="repo-link">GitHub</r-button>
      </div>
    </header>
    <div class="layout">
      <nav class="nav" aria-label="Chapters" id="nav"></nav>
      <main class="article" id="article"></main>
    </div>
  </div>
`;

document.querySelector('#repo-link')?.addEventListener('click', () => {
  window.open('https://github.com/chaxus/arche', '_blank', 'noopener,noreferrer');
});

const nav = document.querySelector('#nav') as HTMLElement;
const article = document.querySelector('#article') as HTMLElement;

function renderNav(activeId: string): void {
  nav.innerHTML = '';
  for (const chapter of CHAPTERS) {
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
  article.innerHTML = `
    <h1>${chapter.title}</h1>
    <p class="lede">${chapter.summary}</p>
    <r-card heading="Lesson" description="ep${chapter.ep}">
      <div class="panel" id="md-host"></div>
    </r-card>
    <div class="math-wrap" id="math-host" hidden></div>
    <p class="footer-note">
      Companion Python:
      <code>python chapters/01-inference/ep${chapter.ep}_*.py</code>
      · Site: <a href="https://arche.chaxus.com">arche.chaxus.com</a>
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

function selectChapter(id: string): void {
  const chapter = CHAPTERS.find((item) => item.id === id) ?? CHAPTERS[0];
  if (!chapter) return;
  const url = new URL(location.href);
  url.searchParams.set('c', chapter.id);
  history.replaceState(null, '', url);
  renderNav(chapter.id);
  renderArticle(chapter);
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
}

selectChapter(initialId);
