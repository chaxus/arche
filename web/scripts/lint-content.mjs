#!/usr/bin/env node
/**
 * Structural lint for beginner curriculum content modules.
 * Usage: node scripts/lint-content.mjs [--strict-template]
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const strict = process.argv.includes('--strict-template');

const PART_IDS = new Set(['start', 'trunk', 'infer', 'train', 'frontier']);

const EN_REQUIRED = [
  'In one sentence',
  'Before you start',
  'The idea (plain language)',
  'In the code',
  'Do this now',
  'If something breaks',
  "Don't worry about yet",
  'Next',
];
const EN_ANY = [...EN_REQUIRED, 'Picture / formula'];

const CN_REQUIRED = [
  '一句话',
  '开始之前',
  '用人话讲',
  '对应代码',
  '现在就做',
  '如果翻车了',
  '先别深挖',
  '下一章',
];
const CN_ANY = [...CN_REQUIRED, '图或公式'];

function readMarkdownLiteral(source, fromIndex) {
  const marker = 'markdown: `';
  const start = source.indexOf(marker, fromIndex);
  if (start < 0 || start - fromIndex > 800) return null;
  const contentStart = start + marker.length;
  // Chapter markdown closes as a line that is exactly "`," (optional indent).
  // Escaped fences inside templates look like \`\`\` and must not match.
  const rest = source.slice(contentStart);
  const m = rest.match(/\n[ \t]*`,/);
  if (!m || m.index == null) return null;
  const tickAt = contentStart + m.index + m[0].lastIndexOf('`');
  return {
    markdown: source.slice(contentStart, tickAt),
    end: contentStart + m.index + m[0].length,
  };
}

function extractChapters(source) {
  const chapters = [];
  const idRe = /\bid:\s*'([^']+)'/g;
  let match;
  while ((match = idRe.exec(source))) {
    const id = match[1];
    const head = source.slice(match.index, match.index + 1200);
    if (!/ep:\s*'/.test(head) || !/partId:\s*'/.test(head)) continue;
    const ep = head.match(/ep:\s*'([^']+)'/)?.[1];
    const title = head.match(/title:\s*'((?:\\'|[^'])*)'/)?.[1];
    const summary = head.match(/summary:\s*'((?:\\'|[^'])*)'/)?.[1];
    const script = head.match(/script:\s*'((?:\\'|[^'])*)'/)?.[1];
    const partId = head.match(/partId:\s*'([^']+)'/)?.[1];
    const estimatedMinutes = Number(head.match(/estimatedMinutes:\s*(\d+)/)?.[1]);
    const md = readMarkdownLiteral(source, match.index);
    if (!ep || !partId || !md || !Number.isFinite(estimatedMinutes)) continue;
    chapters.push({
      id,
      ep,
      title: title ?? '',
      summary: summary ?? '',
      script: script ?? '',
      partId,
      estimatedMinutes,
      markdown: md.markdown,
    });
    idRe.lastIndex = md.end;
  }
  return chapters;
}

function h2s(md) {
  return [...md.matchAll(/^##\s+(.+)$/gm)].map((x) => x[1].trim());
}

function checkTemplate(chapter, required, anyMarkers, label) {
  const heads = h2s(chapter.markdown);
  const hasAny = anyMarkers.some((h) => heads.includes(h));
  if (!hasAny) return [];
  const missing = required.filter((h) => !heads.includes(h));
  return missing.map((h) => `${label} ${chapter.id}: missing ## ${h}`);
}

function loadChapters(files) {
  const chapters = [];
  for (const rel of files) {
    const abs = path.join(root, rel);
    const src = fs.readFileSync(abs, 'utf8');
    chapters.push(...extractChapters(src));
  }
  return chapters;
}

function lintLocale(label, files, required, anyMarkers) {
  const chapters = loadChapters(files);
  const errors = [];
  if (chapters.length === 0) errors.push(`${label}: no chapters parsed`);
  const ids = [];
  const seen = new Set();
  for (const ch of chapters) {
    if (seen.has(ch.id)) errors.push(`${label}: duplicate id ${ch.id}`);
    seen.add(ch.id);
    ids.push(ch.id);
    if (!PART_IDS.has(ch.partId)) errors.push(`${label} ${ch.id}: bad partId ${ch.partId}`);
    if (!Number.isFinite(ch.estimatedMinutes) || ch.estimatedMinutes <= 0) {
      errors.push(`${label} ${ch.id}: bad estimatedMinutes`);
    }
    if (strict) {
      errors.push(...checkTemplate(ch, required, anyMarkers, label));
    }
  }
  return { chapters, errors, ids };
}

const en = lintLocale('EN', ['src/content/part0.ts', 'src/content/en.ts'], EN_REQUIRED, EN_ANY);
const cn = lintLocale('CN', ['src/content/part0-cn.ts', 'src/content/cn.ts'], CN_REQUIRED, CN_ANY);
const errors = [...en.errors, ...cn.errors];

if (en.ids.join() !== cn.ids.join()) {
  errors.push(`id order/set mismatch EN vs CN\nEN: ${en.ids.join(',')}\nCN: ${cn.ids.join(',')}`);
}

if (errors.length) {
  console.error(errors.join('\n'));
  process.exit(1);
}
console.log(
  `ok: ${en.chapters.length} chapters` + (strict ? ' (strict-template)' : ' (structural)'),
);
