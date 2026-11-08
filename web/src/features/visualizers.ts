/** Interactive concept visualizers for RoPE, GQA, and KV cache. */

export type VisualizerKind = 'rope' | 'gqa' | 'kv';

type Copy = {
  hint: string;
  posLabel: string;
  keyLabel: string;
  queryLabel: string;
  kvLabel: string;
  stepLabel: string;
  caption: (state: Record<string, number>) => string;
  noCache?: string;
  withCache?: string;
  showNoCache?: string;
};

const EN: Record<VisualizerKind, Copy> = {
  rope: {
    hint: 'Drag two positions. RoPE rotates Q and K — their relative distance falls out of the angle between them.',
    posLabel: 'Query position',
    keyLabel: 'Key position',
    queryLabel: '',
    kvLabel: '',
    stepLabel: '',
    caption: ({ qPos, kPos, rel, deg }) =>
      `Q@${qPos} · K@${kPos} → Δpos=${rel}, relative angle ≈ ${deg}°. Same Δpos → same Q·K geometry.`,
  },
  gqa: {
    hint: 'Fewer KV heads than query heads: each KV head is shared by a group of queries (Llama 3.2 1B style: 32 Q / 8 KV).',
    posLabel: '',
    keyLabel: '',
    queryLabel: 'Query heads (shown)',
    kvLabel: 'KV heads',
    stepLabel: '',
    caption: ({ q, kv, group }) =>
      `${q} query heads share ${kv} KV heads → group size ${group}. Decode-time cache shrinks.`,
  },
  kv: {
    hint: 'Prefill writes the prompt once. Decode only appends. Toggle the gray “no cache” row to feel the recompute tax.',
    posLabel: '',
    keyLabel: '',
    queryLabel: '',
    kvLabel: '',
    stepLabel: 'Decode step',
    caption: ({ prompt, step, cached, recompute }) =>
      `cache holds ${cached} positions after ${step} new token(s). Without cache you’d redo ~${recompute} attentions over the prompt.`,
    noCache: 'No cache (recompute)',
    withCache: 'With KV cache',
    showNoCache: 'Show no-cache row',
  },
};

const CN: Record<VisualizerKind, Copy> = {
  rope: {
    hint: '拖两个位置。RoPE 旋转 Q 和 K——相对距离会从夹角里读出来。',
    posLabel: 'Query 位置',
    keyLabel: 'Key 位置',
    queryLabel: '',
    kvLabel: '',
    stepLabel: '',
    caption: ({ qPos, kPos, rel, deg }) =>
      `Q@${qPos} · K@${kPos} → Δpos=${rel}，相对角约 ${deg}°。相同 Δpos → 相同 Q·K 几何。`,
  },
  gqa: {
    hint: 'KV 头少于 query 头：每个 KV 头被一组 query 共享（Llama 3.2 1B：32 Q / 8 KV）。',
    posLabel: '',
    keyLabel: '',
    queryLabel: '展示的 Query 头',
    kvLabel: 'KV 头',
    stepLabel: '',
    caption: ({ q, kv, group }) =>
      `${q} 个 query 头共享 ${kv} 个 KV 头 → 每组 ${group} 个。解码时 cache 更小。`,
  },
  kv: {
    hint: 'Prefill 写一次 prompt；Decode 只追加。可开关灰行「无 cache」，对比重算成本。',
    posLabel: '',
    keyLabel: '',
    queryLabel: '',
    kvLabel: '',
    stepLabel: '解码步数',
    caption: ({ prompt, step, cached, recompute }) =>
      `新增 ${step} 个 token 后 cache 有 ${cached} 格；无 cache 大约要重做 ~${recompute} 次对 prompt 的注意力。`,
    noCache: '无 cache（重算）',
    withCache: '有 KV cache',
    showNoCache: '显示无 cache 行',
  },
};

function copyFor(kind: VisualizerKind, localeId: string): Copy {
  return (localeId === 'cn' ? CN : EN)[kind];
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function mountRope(root: HTMLElement, localeId: string): void {
  const copy = copyFor('rope', localeId);
  root.replaceChildren();
  const hint = el('p', 'viz-hint', copy.hint);
  const controls = el('div', 'viz-controls');

  const qRange = el('input') as HTMLInputElement;
  qRange.type = 'range';
  qRange.min = '0';
  qRange.max = '16';
  qRange.value = '3';
  qRange.className = 'viz-range';
  const qLabel = el('label', 'viz-label', copy.posLabel);
  qLabel.append(qRange);

  const kRange = el('input') as HTMLInputElement;
  kRange.type = 'range';
  kRange.min = '0';
  kRange.max = '16';
  kRange.value = '7';
  kRange.className = 'viz-range';
  const kLabel = el('label', 'viz-label', copy.keyLabel);
  kLabel.append(kRange);

  controls.append(qLabel, kLabel);

  const stage = el('div', 'viz-stage');
  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 220;
  canvas.className = 'viz-canvas';
  stage.append(canvas);
  const caption = el('p', 'viz-caption');

  const draw = () => {
    const qPos = Number(qRange.value);
    const kPos = Number(kRange.value);
    const rel = Math.abs(qPos - kPos);
    const deg = (rel * 18) % 360;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    const styles = getComputedStyle(canvas);
    const ink = styles.color || '#111';
    const teal =
      getComputedStyle(document.documentElement).getPropertyValue('--arche-teal').trim() ||
      '#0f6e6a';
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2 + 6;
    const r = 72;
    ctx.strokeStyle = ink;
    ctx.globalAlpha = 0.22;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;

    const qRad = ((qPos * 18) * Math.PI) / 180;
    const kRad = ((kPos * 18) * Math.PI) / 180;

    // K vector
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r * Math.cos(kRad), cy - r * Math.sin(kRad));
    ctx.strokeStyle = ink;
    ctx.globalAlpha = 0.45;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(cx + r * Math.cos(kRad), cy - r * Math.sin(kRad), 4, 0, Math.PI * 2);
    ctx.fillStyle = ink;
    ctx.fill();

    // Q vector (teal)
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r * Math.cos(qRad), cy - r * Math.sin(qRad));
    ctx.strokeStyle = teal;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx + r * Math.cos(qRad), cy - r * Math.sin(qRad), 4.5, 0, Math.PI * 2);
    ctx.fillStyle = teal;
    ctx.fill();

    // Arc between them for relative angle
    const a0 = Math.min(qRad, kRad);
    const a1 = Math.max(qRad, kRad);
    ctx.beginPath();
    ctx.arc(cx, cy, r * 0.42, -a1, -a0);
    ctx.strokeStyle = teal;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;

    ctx.fillStyle = ink;
    ctx.font = '12px IBM Plex Sans, sans-serif';
    ctx.fillText(`Q@${qPos}`, 12, 18);
    ctx.globalAlpha = 0.65;
    ctx.fillText(`K@${kPos}`, 12, 36);
    ctx.globalAlpha = 1;
    caption.textContent = copy.caption({ qPos, kPos, rel, deg });
  };

  qRange.addEventListener('input', draw);
  kRange.addEventListener('input', draw);
  root.append(hint, controls, stage, caption);
  draw();
}

function mountGqa(root: HTMLElement, localeId: string): void {
  const copy = copyFor('gqa', localeId);
  root.replaceChildren();
  const hint = el('p', 'viz-hint', copy.hint);
  const controls = el('div', 'viz-controls');

  const qRange = el('input') as HTMLInputElement;
  qRange.type = 'range';
  qRange.min = '4';
  qRange.max = '16';
  qRange.step = '4';
  qRange.value = '8';
  qRange.className = 'viz-range';
  const qLabel = el('label', 'viz-label', copy.queryLabel);
  qLabel.append(qRange);

  const kvRange = el('input') as HTMLInputElement;
  kvRange.type = 'range';
  kvRange.min = '1';
  kvRange.max = '4';
  kvRange.value = '2';
  kvRange.className = 'viz-range';
  const kvLabel = el('label', 'viz-label', copy.kvLabel);
  kvLabel.append(kvRange);

  controls.append(qLabel, kvLabel);
  const board = el('div', 'viz-gqa-board');
  const caption = el('p', 'viz-caption');

  const render = () => {
    let q = Number(qRange.value);
    let kv = Number(kvRange.value);
    if (q % kv !== 0) {
      while (kv > 1 && q % kv !== 0) kv -= 1;
      kvRange.value = String(kv);
    }
    const group = q / kv;
    board.replaceChildren();
    for (let k = 0; k < kv; k += 1) {
      const col = el('div', 'viz-gqa-col');
      const kvNode = el('div', 'viz-gqa-kv', `KV ${k}`);
      col.append(kvNode);
      const links = el('div', 'viz-gqa-links');
      for (let g = 0; g < group; g += 1) {
        const qi = k * group + g;
        const link = el('div', 'viz-gqa-link');
        link.append(el('span', 'viz-gqa-wire', ''), el('div', 'viz-gqa-q', `Q ${qi}`));
        links.append(link);
      }
      col.append(links);
      board.append(col);
    }
    caption.textContent = copy.caption({ q, kv, group });
  };

  qRange.addEventListener('input', render);
  kvRange.addEventListener('input', render);
  root.append(hint, controls, board, caption);
  render();
}

function mountKv(root: HTMLElement, localeId: string): void {
  const copy = copyFor('kv', localeId);
  root.replaceChildren();
  const hint = el('p', 'viz-hint', copy.hint);
  const promptLen = 5;
  const controls = el('div', 'viz-controls');
  const range = el('input') as HTMLInputElement;
  range.type = 'range';
  range.min = '0';
  range.max = '6';
  range.value = '2';
  range.className = 'viz-range';
  const label = el('label', 'viz-label', copy.stepLabel);
  label.append(range);

  const toggle = el('label', 'viz-label viz-toggle');
  const check = el('input') as HTMLInputElement;
  check.type = 'checkbox';
  check.checked = true;
  toggle.append(check, document.createTextNode(` ${copy.showNoCache ?? ''}`));
  controls.append(label, toggle);

  const withLabel = el('p', 'viz-row-label', copy.withCache ?? '');
  const row = el('div', 'viz-kv-row');
  const noLabel = el('p', 'viz-row-label is-muted', copy.noCache ?? '');
  const noRow = el('div', 'viz-kv-row is-nocache');
  const caption = el('p', 'viz-caption');

  const render = () => {
    const step = Number(range.value);
    const showNoCache = check.checked;
    row.replaceChildren();
    noRow.replaceChildren();
    for (let i = 0; i < promptLen; i += 1) {
      const cell = el('div', 'viz-kv-cell prefill', `P${i}`);
      cell.style.setProperty('--i', String(i));
      row.append(cell);
      if (showNoCache) noRow.append(el('div', 'viz-kv-cell prefill is-ghost', `P${i}`));
    }
    for (let i = 0; i < step; i += 1) {
      const cell = el('div', 'viz-kv-cell decode', `D${i}`);
      cell.style.setProperty('--i', String(promptLen + i));
      row.append(cell);
      if (showNoCache) noRow.append(el('div', 'viz-kv-cell decode is-ghost', `D${i}`));
    }
    noLabel.hidden = !showNoCache;
    noRow.hidden = !showNoCache;
    // Without cache: each decode step re-attends over the whole prompt + earlier tokens.
    const recompute = promptLen * Math.max(step, 1);
    caption.textContent = copy.caption({
      prompt: promptLen,
      step,
      cached: promptLen + step,
      recompute,
    });
  };

  range.addEventListener('input', render);
  check.addEventListener('change', render);
  root.append(hint, controls, withLabel, row, noLabel, noRow, caption);
  render();
}

export function mountVisualizer(
  host: HTMLElement,
  kind: VisualizerKind,
  localeId: string,
): void {
  host.hidden = false;
  const wrap = el('div', 'viz-wrap');
  if (kind === 'rope') mountRope(wrap, localeId);
  else if (kind === 'gqa') mountGqa(wrap, localeId);
  else mountKv(wrap, localeId);
  host.replaceChildren(wrap);
}
