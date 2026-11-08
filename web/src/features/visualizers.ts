/** Interactive concept visualizers for RoPE, GQA, and KV cache. */

export type VisualizerKind = 'rope' | 'gqa' | 'kv';

type Copy = {
  hint: string;
  posLabel: string;
  queryLabel: string;
  kvLabel: string;
  stepLabel: string;
  caption: (state: Record<string, number>) => string;
};

const EN: Record<VisualizerKind, Copy> = {
  rope: {
    hint: 'Drag position. RoPE rotates the same vector to a new angle — relative position falls out of the dot product.',
    posLabel: 'Token position',
    queryLabel: '',
    kvLabel: '',
    stepLabel: '',
    caption: ({ pos, deg }) =>
      `position = ${pos} → rotation ≈ ${deg}°. Neighboring tokens get nearby angles.`,
  },
  gqa: {
    hint: 'Fewer KV heads than query heads: each KV head is shared by a group of queries (Llama 3.2 1B style: 32 Q / 8 KV).',
    posLabel: '',
    queryLabel: 'Query heads (shown)',
    kvLabel: 'KV heads',
    stepLabel: '',
    caption: ({ q, kv, group }) =>
      `${q} query heads share ${kv} KV heads → group size ${group}. Decode-time cache shrinks.`,
  },
  kv: {
    hint: 'Prefill writes the prompt into the cache once. Each decode step only appends the new token’s K/V.',
    posLabel: '',
    queryLabel: '',
    kvLabel: '',
    stepLabel: 'Decode step',
    caption: ({ prompt, step, cached }) =>
      `prompt length ${prompt}; after ${step} new token(s) the cache holds ${cached} positions.`,
  },
};

const CN: Record<VisualizerKind, Copy> = {
  rope: {
    hint: '拖动位置。RoPE 把同一个向量转到新角度——相对位置会从点积里出现。',
    posLabel: 'Token 位置',
    queryLabel: '',
    kvLabel: '',
    stepLabel: '',
    caption: ({ pos, deg }) => `位置 = ${pos} → 旋转约 ${deg}°。相邻 token 的角度也相邻。`,
  },
  gqa: {
    hint: 'KV 头少于 query 头：每个 KV 头被一组 query 共享（Llama 3.2 1B：32 Q / 8 KV）。',
    posLabel: '',
    queryLabel: '展示的 Query 头',
    kvLabel: 'KV 头',
    stepLabel: '',
    caption: ({ q, kv, group }) =>
      `${q} 个 query 头共享 ${kv} 个 KV 头 → 每组 ${group} 个。解码时 cache 更小。`,
  },
  kv: {
    hint: 'Prefill 把 prompt 写入 cache 一次；之后每解码一步只追加新 token 的 K/V。',
    posLabel: '',
    queryLabel: '',
    kvLabel: '',
    stepLabel: '解码步数',
    caption: ({ prompt, step, cached }) =>
      `prompt 长 ${prompt}；新增 ${step} 个 token 后，cache 里有 ${cached} 个位置。`,
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
  const label = el('label', 'viz-label', copy.posLabel);
  const range = el('input') as HTMLInputElement;
  range.type = 'range';
  range.min = '0';
  range.max = '16';
  range.value = '3';
  range.className = 'viz-range';
  label.append(range);
  controls.append(label);

  const stage = el('div', 'viz-stage');
  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 200;
  canvas.className = 'viz-canvas';
  stage.append(canvas);
  const caption = el('p', 'viz-caption');

  const draw = () => {
    const pos = Number(range.value);
    const deg = (pos * 18) % 360;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2;
    const r = 70;
    ctx.strokeStyle = 'currentColor';
    ctx.globalAlpha = 0.25;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
    // baseline at pos 0
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r, cy);
    ctx.strokeStyle = 'currentColor';
    ctx.globalAlpha = 0.35;
    ctx.stroke();
    ctx.globalAlpha = 1;
    const rad = (deg * Math.PI) / 180;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r * Math.cos(rad), cy - r * Math.sin(rad));
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx + r * Math.cos(rad), cy - r * Math.sin(rad), 4, 0, Math.PI * 2);
    ctx.fillStyle = 'currentColor';
    ctx.fill();
    caption.textContent = copy.caption({ pos, deg });
  };

  range.addEventListener('input', draw);
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
      // snap kv down to a divisor
      while (kv > 1 && q % kv !== 0) kv -= 1;
      kvRange.value = String(kv);
    }
    const group = q / kv;
    board.replaceChildren();
    for (let k = 0; k < kv; k += 1) {
      const col = el('div', 'viz-gqa-col');
      const kvNode = el('div', 'viz-gqa-kv', `KV ${k}`);
      col.append(kvNode);
      for (let g = 0; g < group; g += 1) {
        const qi = k * group + g;
        col.append(el('div', 'viz-gqa-q', `Q ${qi}`));
      }
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
  range.value = '0';
  range.className = 'viz-range';
  const label = el('label', 'viz-label', copy.stepLabel);
  label.append(range);
  controls.append(label);

  const row = el('div', 'viz-kv-row');
  const caption = el('p', 'viz-caption');

  const render = () => {
    const step = Number(range.value);
    row.replaceChildren();
    for (let i = 0; i < promptLen; i += 1) {
      const cell = el('div', 'viz-kv-cell prefill', `P${i}`);
      row.append(cell);
    }
    for (let i = 0; i < step; i += 1) {
      const cell = el('div', 'viz-kv-cell decode', `D${i}`);
      row.append(cell);
    }
    caption.textContent = copy.caption({
      prompt: promptLen,
      step,
      cached: promptLen + step,
    });
  };

  range.addEventListener('input', render);
  root.append(hint, controls, row, caption);
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
