#!/usr/bin/env node
/**
 * Structural lint for web/src/content/{en,cn}.ts
 * Usage: node scripts/lint-content.mjs [--strict-template]
 *
 * Without --strict-template: require id, partId, estimatedMinutes, matching ids.
 * With --strict-template: if a chapter already has any template H2, require the full set.
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

function extractChapters(source) {
  const chapters = [];
  const re =
    /\{\s*id:\s*'([^']+)'[\s\S]*?ep:\s*'([^']+)'[\s\S]*?title:\s*'((?:\\'|[^'])*)'[\s\S]*?summary:\s*'((?:\\'|[^'])*)'[\s\S]*?script:\s*'((?:\\'|[^'])*)'[\s\S]*?partId:\s*'([^']+)'[\s\S]*?estimatedMinutes:\s*(\d+)[\s\S]*?markdown:\s*`([\s\S]*?)`/g;
  let m;
  while ((m = re.exec(source))) {
    chapters.push({
      id: m[1],
      ep: m[2],
      title: m[3],
      summary: m[4],
      script: m[5],
      partId: m[6],
      estimatedMinutes: Number(m[7]),
      markdown: m[8],
    });
  }
  return chapters;
}

function h2s(md) {
  return [...md.matchAll(/^##\s+(.+)$/gm)].map((x) => x[1].trim());
}

function checkTemplate(chapter, required, anyMarkers, label) {
  const heads = h2s(chapter.markdown);
  const hasAny = anyMarkers.some((h) => heads.includes(h));
  if (!hasAny) return []; // legacy chapter
  const missing = required.filter((h) => !heads.includes(h));
  return missing.map((h) => `${label} ${chapter.id}: missing ## ${h}`);
}

function lintFile(rel, required, anyMarkers) {
  const abs = path.join(root, rel);
  const src = fs.readFileSync(abs, 'utf8');
  const chapters = extractChapters(src);
  const errors = [];
  if (chapters.length === 0) errors.push(`${rel}: no chapters parsed`);
  const ids = new Set();
  for (const ch of chapters) {
    if (ids.has(ch.id)) errors.push(`${rel}: duplicate id ${ch.id}`);
    ids.add(ch.id);
    if (!PART_IDS.has(ch.partId)) errors.push(`${rel} ${ch.id}: bad partId ${ch.partId}`);
    if (!Number.isFinite(ch.estimatedMinutes) || ch.estimatedMinutes <= 0) {
      errors.push(`${rel} ${ch.id}: bad estimatedMinutes`);
    }
    if (strict) {
      errors.push(...checkTemplate(ch, required, anyMarkers, rel));
    }
  }
  return { chapters, errors, ids: [...ids] };
}

const en = lintFile('src/content/en.ts', EN_REQUIRED, EN_ANY);
const cn = lintFile('src/content/cn.ts', CN_REQUIRED, CN_ANY);
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
