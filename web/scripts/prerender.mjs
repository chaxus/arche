/**
 * Post-Vite prerender — same idea as ran docs: one URL map → head + static HTML files.
 * Emits path/index.html, sitemap.xml, llms-full.txt. robots.txt / llms.txt stay in public/.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, '..');
const distDir = path.join(webRoot, 'dist');
const contentDir = path.join(webRoot, 'src', 'content');

const ORIGIN = 'https://arche.chaxus.com';
const SITE = {
  name: 'arche',
  github: 'https://github.com/chaxus/arche',
  descriptionEn:
    'From zero toward frontier-ready: hand-write a Llama 3.2–style LLM, train or load weights, serve faster, then map MoE/MLA.',
  descriptionCn:
    '从零对接前沿：亲手写出 Llama 3.2 风格大模型主干，训练或加载权重，服务加速，再摸 MoE/MLA 地图。',
};

const LOCALES = [
  {
    id: 'root',
    dir: '',
    lang: 'en',
    hreflang: 'en',
    ogLocale: 'en_US',
    label: 'English',
    files: ['part0.ts', 'en.ts'],
    mdDir: 'beginner/en',
    homeTitle: 'arche — build an LLM by hand, from zero',
    homeHeading: 'See what an LLM is doing—by writing one',
    homeLede: SITE.descriptionEn,
  },
  {
    id: 'cn',
    dir: 'cn',
    lang: 'zh-CN',
    hreflang: 'zh-CN',
    ogLocale: 'zh_CN',
    label: '简体中文',
    files: ['part0-cn.ts', 'cn.ts'],
    mdDir: 'beginner/cn',
    homeTitle: 'arche — 从零开始，亲手写出大模型',
    homeHeading: '亲手写出一个大模型，看懂它在算什么',
    homeLede: SITE.descriptionCn,
  },
];

const escapeHtml = (value) =>
  String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');

const absoluteUrl = (pathname) => `${ORIGIN}${pathname.startsWith('/') ? pathname : `/${pathname}`}`;

const pagePath = (chapterId, locale) => {
  if (chapterId === 'home') return locale.dir ? `/${locale.dir}/` : '/';
  const base = locale.dir ? `/${locale.dir}/${chapterId}/` : `/${chapterId}/`;
  return base;
};

const unescapeTs = (value) =>
  value
    .replaceAll("\\'", "'")
    .replaceAll('\\`', '`')
    .replaceAll('\\n', '\n')
    .replaceAll('\\$', '$')
    .replaceAll('\\\\', '\\');

/** Pull chapter records from content TS (id / title / summary / markdown). */
function parseChaptersFromSource(source) {
  const chapters = [];
  const idRe = /\bid:\s*'([^']+)'/g;
  let match;
  while ((match = idRe.exec(source))) {
    const id = match[1];
    if (id === 'home') continue;
    const slice = source.slice(match.index, match.index + 12000);
    const titleM = slice.match(/\btitle:\s*'((?:\\'|[^'])*)'/);
    const summaryM = slice.match(/\bsummary:\s*'((?:\\'|[^'])*)'/);
    const epM = slice.match(/\bep:\s*'((?:\\'|[^'])*)'/);
    if (!titleM || !summaryM) continue;

    let markdown = '';
    const mdKey = slice.indexOf('markdown:');
    if (mdKey >= 0) {
      const after = slice.slice(mdKey + 'markdown:'.length).trimStart();
      if (after.startsWith('`')) {
        let i = 1;
        let out = '';
        while (i < after.length) {
          if (after[i] === '\\' && i + 1 < after.length) {
            out += after[i + 1];
            i += 2;
            continue;
          }
          if (after[i] === '`') break;
          out += after[i];
          i += 1;
        }
        markdown = out;
      }
    }

    chapters.push({
      id,
      ep: epM ? unescapeTs(epM[1]) : '',
      title: unescapeTs(titleM[1]),
      summary: unescapeTs(summaryM[1]),
      markdown,
    });
  }
  return chapters;
}

function loadLocaleChapters(locale) {
  const seen = new Set();
  const chapters = [];
  for (const file of locale.files) {
    const source = fs.readFileSync(path.join(contentDir, file), 'utf8');
    for (const chapter of parseChaptersFromSource(source)) {
      if (seen.has(chapter.id)) continue;
      seen.add(chapter.id);
      const mdPath = path.join(contentDir, locale.mdDir, `${chapter.id}.md`);
      if (fs.existsSync(mdPath)) {
        chapter.markdown = fs.readFileSync(mdPath, 'utf8');
      }
      chapters.push(chapter);
    }
  }
  return chapters;
}

/** Minimal markdown → HTML for crawler-visible body (not a full CommonMark port). */
function mdToHtml(md) {
  const lines = md.replaceAll('\r\n', '\n').split('\n');
  const out = [];
  let inCode = false;
  let codeLang = '';
  let codeBuf = [];
  let para = [];

  const flushPara = () => {
    if (!para.length) return;
    out.push(`<p>${escapeHtml(para.join(' ')).replaceAll(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')}</p>`);
    para = [];
  };

  for (const line of lines) {
    const fence = line.match(/^```(\w*)/);
    if (fence) {
      if (inCode) {
        out.push(
          `<pre><code${codeLang ? ` class="language-${escapeHtml(codeLang)}"` : ''}>${escapeHtml(codeBuf.join('\n'))}</code></pre>`,
        );
        codeBuf = [];
        inCode = false;
        codeLang = '';
      } else {
        flushPara();
        inCode = true;
        codeLang = fence[1] || '';
      }
      continue;
    }
    if (inCode) {
      codeBuf.push(line);
      continue;
    }
    if (!line.trim()) {
      flushPara();
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushPara();
      const level = heading[1].length;
      out.push(`<h${level}>${escapeHtml(heading[2]).replaceAll(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')}</h${level}>`);
      continue;
    }
    if (/^\|/.test(line)) {
      flushPara();
      out.push(`<pre class="ssg-table">${escapeHtml(line)}</pre>`);
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      flushPara();
      out.push(`<li>${escapeHtml(line.replace(/^[-*]\s+/, '')).replaceAll(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')}</li>`);
      continue;
    }
    para.push(line.trim());
  }
  flushPara();
  if (inCode) {
    out.push(`<pre><code>${escapeHtml(codeBuf.join('\n'))}</code></pre>`);
  }
  return out.join('\n');
}

function headTags({ locale, chapterId, title, description, path: pathname }) {
  const url = absoluteUrl(pathname);
  const tags = [
    `<meta charset="UTF-8" />`,
    `<meta name="viewport" content="width=device-width, initial-scale=1" />`,
    `<title>${escapeHtml(title)}</title>`,
    `<meta name="description" content="${escapeHtml(description)}" />`,
    `<meta name="author" content="chaxus" />`,
    `<link rel="canonical" href="${url}" />`,
  ];

  for (const other of LOCALES) {
    const href = absoluteUrl(pagePath(chapterId, other));
    tags.push(`<link rel="alternate" hreflang="${other.hreflang}" href="${href}" />`);
  }
  tags.push(`<link rel="alternate" hreflang="x-default" href="${absoluteUrl(pagePath(chapterId, LOCALES[0]))}" />`);
  tags.push(`<link rel="alternate" type="text/plain" title="llms.txt" href="${ORIGIN}/llms.txt" />`);

  tags.push(
    `<meta property="og:type" content="website" />`,
    `<meta property="og:site_name" content="${SITE.name}" />`,
    `<meta property="og:title" content="${escapeHtml(title)}" />`,
    `<meta property="og:description" content="${escapeHtml(description)}" />`,
    `<meta property="og:url" content="${url}" />`,
    `<meta property="og:locale" content="${locale.ogLocale}" />`,
    `<meta name="twitter:card" content="summary" />`,
    `<meta name="twitter:title" content="${escapeHtml(title)}" />`,
    `<meta name="twitter:description" content="${escapeHtml(description)}" />`,
  );

  if (chapterId === 'home') {
    const jsonLd = {
      '@context': 'https://schema.org',
      '@type': 'WebSite',
      name: SITE.name,
      url: ORIGIN,
      description,
      author: { '@type': 'Person', name: 'chaxus', url: 'https://chaxus.com/' },
      inLanguage: [LOCALES[0].lang, LOCALES[1].lang],
    };
    tags.push(`<script type="application/ld+json">${JSON.stringify(jsonLd)}</script>`);
  } else {
    const jsonLd = {
      '@context': 'https://schema.org',
      '@type': 'LearningResource',
      name: title,
      description,
      url,
      isPartOf: { '@type': 'WebSite', name: SITE.name, url: ORIGIN },
      inLanguage: locale.lang,
      learningResourceType: 'chapter',
    };
    tags.push(`<script type="application/ld+json">${JSON.stringify(jsonLd)}</script>`);
  }

  return tags;
}

function writePage(relPath, html) {
  const outPath = path.join(distDir, relPath);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, html);
}

function buildHtml({ locale, chapterId, title, description, bodyHtml, assetTags }) {
  const pathname = pagePath(chapterId, locale);
  const head = [...headTags({ locale, chapterId, title, description, path: pathname }), ...assetTags].join(
    '\n    ',
  );

  return `<!doctype html>
<html lang="${locale.lang}">
  <head>
    ${head}
  </head>
  <body>
    <div id="app">
      <article class="ssg-article" data-chapter="${escapeHtml(chapterId)}">
        <p class="ssg-brand"><a href="${pagePath('home', locale)}">arche</a></p>
        <h1>${escapeHtml(title.replace(/^arche — /, '').replace(/^arche — /, ''))}</h1>
        <p class="ssg-lede">${escapeHtml(description)}</p>
        ${bodyHtml}
        <p class="ssg-note">Interactive lesson loads when JavaScript is available.</p>
      </article>
    </div>
  </body>
</html>
`;
}

function extractAssetTags(indexHtml) {
  const tags = [];
  for (const match of indexHtml.matchAll(/<script type="module"[^>]*src="([^"]+)"[^>]*><\/script>/g)) {
    tags.push(`<script type="module" crossorigin src="${match[1]}"></script>`);
  }
  for (const match of indexHtml.matchAll(/<link rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g)) {
    tags.push(`<link rel="stylesheet" crossorigin href="${match[1]}">`);
  }
  // Vite may emit modulepreload
  for (const match of indexHtml.matchAll(/<link rel="modulepreload"[^>]*>/g)) {
    tags.push(match[0]);
  }
  return tags;
}

function renderSitemap(urls) {
  const body = urls
    .map(
      (item) => `  <url>
    <loc>${item.loc}</loc>
    <priority>${item.priority}</priority>
  </url>`,
    )
    .join('\n');
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${body}
</urlset>
`;
}

function main() {
  if (!fs.existsSync(distDir)) {
    console.error('dist/ missing — run vite build first');
    process.exit(1);
  }

  const indexHtml = fs.readFileSync(path.join(distDir, 'index.html'), 'utf8');
  const assetTags = extractAssetTags(indexHtml);
  const sitemapUrls = [];
  const llmsLines = [
    '# arche',
    '',
    `> ${SITE.descriptionEn}`,
    '',
    `GitHub: ${SITE.github}`,
    `Site: ${ORIGIN}`,
    '',
    '## Docs',
    '',
  ];
  const llmsFull = [
    '# arche — full curriculum text for AI systems',
    '',
    `> ${SITE.descriptionEn}`,
    '',
  ];

  for (const locale of LOCALES) {
    const chapters = loadLocaleChapters(locale);
    const homePath = pagePath('home', locale);
    const homeTitle = locale.homeTitle;
    const homeDesc = locale.homeLede;

    const homeLinks = chapters
      .map((ch) => `<li><a href="${pagePath(ch.id, locale)}">${escapeHtml(ch.title)}</a> — ${escapeHtml(ch.summary)}</li>`)
      .join('\n');

    const homeHtml = buildHtml({
      locale,
      chapterId: 'home',
      title: homeTitle,
      description: homeDesc,
      bodyHtml: `<p>${escapeHtml(locale.homeHeading)}</p><ol class="ssg-toc">${homeLinks}</ol>`,
      assetTags,
    });

    if (locale.dir) {
      writePage(path.join(locale.dir, 'index.html'), homeHtml);
    } else {
      writePage('index.html', homeHtml);
    }
    sitemapUrls.push({ loc: absoluteUrl(homePath), priority: '1.0' });
    llmsLines.push(
      `- [Home (${locale.label})](${absoluteUrl(homePath)}): ${homeDesc}`,
    );

    for (const chapter of chapters) {
      const pathname = pagePath(chapter.id, locale);
      const title = `${chapter.title} — arche`;
      const description = chapter.summary;
      const bodyHtml = chapter.markdown
        ? mdToHtml(chapter.markdown)
        : `<p>${escapeHtml(chapter.summary)}</p>`;

      const html = buildHtml({
        locale,
        chapterId: chapter.id,
        title,
        description,
        bodyHtml,
        assetTags,
      });
      const rel =
        locale.dir != null && locale.dir !== ''
          ? path.join(locale.dir, chapter.id, 'index.html')
          : path.join(chapter.id, 'index.html');
      writePage(rel, html);
      sitemapUrls.push({ loc: absoluteUrl(pathname), priority: '0.7' });
      llmsLines.push(
        `- [${chapter.title} (${locale.label})](${absoluteUrl(pathname)}): ${chapter.summary}`,
      );
      llmsFull.push(`## ${chapter.title} (${locale.label})`);
      llmsFull.push('');
      llmsFull.push(`URL: ${absoluteUrl(pathname)}`);
      llmsFull.push('');
      llmsFull.push(chapter.markdown || chapter.summary);
      llmsFull.push('');
    }
  }

  fs.writeFileSync(path.join(distDir, 'sitemap.xml'), renderSitemap(sitemapUrls));
  fs.writeFileSync(path.join(distDir, 'llms-full.txt'), `${llmsFull.join('\n')}\n`);

  // Augment hand-maintained llms.txt copied from public/ with chapter index if present
  const publicLlms = path.join(webRoot, 'public', 'llms.txt');
  if (fs.existsSync(publicLlms)) {
    const base = fs.readFileSync(publicLlms, 'utf8').trimEnd();
    const generated = `\n\n## Generated chapter index\n\n${llmsLines.slice(8).join('\n')}\n`;
    fs.writeFileSync(path.join(distDir, 'llms.txt'), `${base}${generated}`);
  } else {
    fs.writeFileSync(path.join(distDir, 'llms.txt'), `${llmsLines.join('\n')}\n`);
  }

  let missing = 0;
  for (const item of sitemapUrls) {
    const u = new URL(item.loc);
    const parts = u.pathname.split('/').filter(Boolean);
    const filePath =
      parts.length === 0
        ? path.join(distDir, 'index.html')
        : path.join(distDir, ...parts, 'index.html');
    if (!fs.existsSync(filePath)) {
      missing += 1;
      console.warn('missing page for', item.loc);
    }
  }
  if (missing) {
    console.error(`prerender verify failed: ${missing} missing pages`);
    process.exit(1);
  }

  console.log(
    `prerender: ${sitemapUrls.length} urls, sitemap.xml, llms.txt, llms-full.txt`,
  );
}

main();
