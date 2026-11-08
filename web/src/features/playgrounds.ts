/** In-browser teaching playgrounds (not full PyTorch / Llama weights). */

export type PlaygroundLocale = 'en' | 'cn';

type Labels = {
  heading: string;
  hint: string;
  encode: string;
  note: string;
  inputPh: string;
  tokens: string;
  ids: string;
  decoded: string;
  lr: string;
  steps: string;
  train: string;
  lossTitle: string;
};

type EmbLabels = {
  heading: string;
  hint: string;
  lookup: string;
  note: string;
  inputPh: string;
  table: string;
  gathered: string;
  shape: string;
};

const LABELS: Record<
  PlaygroundLocale,
  { tokenizer: Labels; embedding: EmbLabels; pretrain: Labels }
> = {
  en: {
    tokenizer: {
      heading: 'Try it here',
      hint: 'Type text → split with a tiny teaching BPE → see tokens and ids. The lab script below still uses official Llama 3.2.',
      encode: 'Encode',
      note: 'UTF-8 bytes + a few English merges (teaching table ≠ Llama). Encode then decode should match your input.',
      inputPh: 'The cat sat on the mat.',
      tokens: 'Tokens',
      ids: 'Ids',
      decoded: 'Decoded',
      lr: 'Learning rate',
      steps: 'Steps',
      train: 'Train',
      lossTitle: 'Loss',
    },
    embedding: {
      heading: 'Try it here',
      hint: 'Type ids like 3, 7, 11. Embedding only copies those rows from the table—no “understanding”.',
      lookup: 'Copy rows',
      note: 'Tiny fake table (12 rows × 4 floats). Real Llama is huge; the action is still copy-by-id.',
      inputPh: '3, 7, 11',
      table: 'The table (each id has one row)',
      gathered: 'Copied rows (what later layers see)',
      shape: 'How many rows you copied',
    },
    pretrain: {
      heading: 'Try it here',
      hint: 'Tiny next-token model in the browser. Drag learning rate and steps, then train — watch loss move.',
      encode: '',
      note: 'Toy vocab=32, dim=8, one embed + linear. Same API shape as the chapter script, not a real 1B run.',
      inputPh: '',
      tokens: '',
      ids: '',
      decoded: '',
      lr: 'Learning rate',
      steps: 'Steps',
      train: 'Train',
      lossTitle: 'Loss curve',
    },
  },
  cn: {
    tokenizer: {
      heading: '在页面上试',
      hint: '输入文字 → 用教学用小 BPE 切开 → 看 token 和 id。下方实验脚本仍用官方 Llama 3.2。',
      encode: '编码',
      note: '按 UTF-8 字节起步 + 少量英文合并（教学表 ≠ Llama）。编码再解码应能还原原文。',
      inputPh: 'The cat sat on the mat.',
      tokens: 'Tokens',
      ids: '编号',
      decoded: '解码回来',
      lr: '学习率',
      steps: '步数',
      train: '训练',
      lossTitle: '损失',
    },
    embedding: {
      heading: '在页面上试',
      hint: '输入编号，例如 3, 7, 11。嵌入只会把表里对应的行抄出来——没有「理解句子」。',
      lookup: '按编号抄行',
      note: '教学假表（12 行 × 每行 4 个小数）。真 Llama 大很多；动作仍是按 id 抄行。',
      inputPh: '3, 7, 11',
      table: '那张表（每个 id 一行）',
      gathered: '抄出来的行（后面层只看到这些小数）',
      shape: '抄了几行',
    },
    pretrain: {
      heading: '在页面上试',
      hint: '浏览器里的玩具 next-token 模型。拖学习率和步数，点训练，看 loss 变化。',
      encode: '',
      note: '词表 32、维度 8、一层嵌入 + 线性。和章节脚本同一类 API，不是真训 1B。',
      inputPh: '',
      tokens: '',
      ids: '',
      decoded: '',
      lr: '学习率',
      steps: '步数',
      train: '训练',
      lossTitle: 'Loss 曲线',
    },
  },
};

/** Teaching BPE on UTF-8 bytes (same idea as production). Merges are ASCII demos. */
const MERGE_STRINGS: [string, string][] = [
  ['t', 'h'],
  ['h', 'e'],
  ['i', 'n'],
  ['e', 'r'],
  ['a', 'n'],
  ['r', 'e'],
  ['o', 'n'],
  ['e', 'n'],
  ['a', 't'],
  ['e', 'd'],
  ['i', 's'],
  ['o', 'r'],
  ['i', 't'],
  ['a', 'l'],
  ['a', 'r'],
  ['s', 't'],
  ['e', 's'],
  ['n', 'g'],
  ['o', 'u'],
  ['i', 'o'],
  ['l', 'e'],
  ['v', 'e'],
  ['c', 'o'],
  ['m', 'e'],
  ['c', 'a'],
  ['th', 'e'],
  ['the', ' '],
  ['i', 'ng'],
  ['on', ' '],
  ['at', ' '],
];

const enc = new TextEncoder();
const dec = new TextDecoder();

type Piece = Uint8Array;

function bytesKey(p: Piece): string {
  return Array.from(p).join(',');
}

function strToBytes(s: string): Piece {
  return enc.encode(s);
}

function bytesToDisplay(p: Piece): string {
  try {
    return dec.decode(p);
  } catch {
    return Array.from(p)
      .map((b) => `\\x${b.toString(16).padStart(2, '0')}`)
      .join('');
  }
}

function concatBytes(a: Piece, b: Piece): Piece {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function eqBytes(a: Piece, b: Piece): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
  return true;
}

function buildTeachVocab(): {
  pieceToId: Map<string, number>;
  idToPiece: Piece[];
  merges: [Piece, Piece][];
} {
  const idToPiece: Piece[] = [];
  const pieceToId = new Map<string, number>();
  const add = (p: Piece) => {
    const k = bytesKey(p);
    if (pieceToId.has(k)) return;
    pieceToId.set(k, idToPiece.length);
    idToPiece.push(p);
  };
  for (let b = 0; b < 256; b += 1) add(Uint8Array.of(b));
  const merges: [Piece, Piece][] = MERGE_STRINGS.map(([a, b]) => {
    const left = strToBytes(a);
    const right = strToBytes(b);
    add(concatBytes(left, right));
    return [left, right];
  });
  return { pieceToId, idToPiece, merges };
}

const TEACH = buildTeachVocab();

export function teachBpeEncode(text: string): { tokens: string[]; ids: number[] } {
  let parts: Piece[] = [...enc.encode(text)].map((b) => Uint8Array.of(b));
  for (const [a, b] of TEACH.merges) {
    const next: Piece[] = [];
    for (let i = 0; i < parts.length; i += 1) {
      if (i + 1 < parts.length && eqBytes(parts[i]!, a) && eqBytes(parts[i + 1]!, b)) {
        next.push(concatBytes(a, b));
        i += 1;
      } else {
        next.push(parts[i]!);
      }
    }
    parts = next;
  }
  const ids = parts.map((p) => TEACH.pieceToId.get(bytesKey(p)) ?? 0);
  const tokens = parts.map(bytesToDisplay);
  return { tokens, ids };
}

export function teachBpeDecode(ids: number[]): string {
  const bytes: number[] = [];
  for (const id of ids) {
    const piece = TEACH.idToPiece[id];
    if (!piece) continue;
    for (const b of piece) bytes.push(b);
  }
  return dec.decode(Uint8Array.from(bytes));
}

function escapeHtml(s: string): string {
  return s
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function showToken(t: string): string {
  if (t === ' ') return '␠';
  if (t === '\n') return '\\n';
  return t;
}

export function mountTokenizerPlayground(host: HTMLElement, locale: PlaygroundLocale): void {
  const L = LABELS[locale].tokenizer;
  host.hidden = false;
  host.innerHTML = `
    <r-card heading="${escapeHtml(L.heading)}" description="tokenizer">
      <p class="demo-hint">${escapeHtml(L.hint)}</p>
      <textarea class="pg-input" id="pg-tok-in" rows="3" spellcheck="false">${escapeHtml(L.inputPh)}</textarea>
      <div class="pg-actions">
        <r-button type="primary" id="pg-tok-run">${escapeHtml(L.encode)}</r-button>
      </div>
      <p class="pg-note">${escapeHtml(L.note)}</p>
      <div class="pg-block"><strong>${escapeHtml(L.tokens)}</strong><div class="pg-chips" id="pg-tok-chips"></div></div>
      <div class="pg-block"><strong>${escapeHtml(L.ids)}</strong><pre class="demo-out" id="pg-tok-ids"></pre></div>
      <div class="pg-block"><strong>${escapeHtml(L.decoded)}</strong><pre class="demo-out" id="pg-tok-dec"></pre></div>
    </r-card>
  `;

  const paint = () => {
    const text = (host.querySelector('#pg-tok-in') as HTMLTextAreaElement).value;
    const { tokens, ids } = teachBpeEncode(text);
    const chips = host.querySelector('#pg-tok-chips')!;
    chips.innerHTML = tokens
      .map((t) => `<span class="pg-chip">${escapeHtml(showToken(t))}</span>`)
      .join('');
    (host.querySelector('#pg-tok-ids') as HTMLElement).textContent = `[${ids.join(', ')}]  (${ids.length})`;
    (host.querySelector('#pg-tok-dec') as HTMLElement).textContent = JSON.stringify(teachBpeDecode(ids));
  };

  host.querySelector('#pg-tok-run')?.addEventListener('click', paint);
  paint();
}

/** Fixed tiny embedding table for teaching (vocab=12, dim=4). */
const TEACH_EMB_VOCAB = 12;
const TEACH_EMB_DIM = 4;
const TEACH_EMB: number[][] = (() => {
  let s = 42 >>> 0;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
  return Array.from({ length: TEACH_EMB_VOCAB }, () =>
    Array.from({ length: TEACH_EMB_DIM }, () => Number((rand() * 2 - 1).toFixed(2))),
  );
})();

function parseIds(raw: string, vocab: number): { ids: number[]; error?: string } {
  const parts = raw
    .split(/[,，\s]+/)
    .map((p) => p.trim())
    .filter(Boolean);
  if (parts.length === 0) return { ids: [], error: 'empty' };
  const ids: number[] = [];
  for (const p of parts) {
    if (!/^\d+$/.test(p)) return { ids: [], error: p };
    const n = Number(p);
    if (n < 0 || n >= vocab) return { ids: [], error: String(n) };
    ids.push(n);
  }
  return { ids };
}

export function mountEmbeddingPlayground(host: HTMLElement, locale: PlaygroundLocale): void {
  const L = LABELS[locale].embedding;
  host.hidden = false;
  host.innerHTML = `
    <r-card heading="${escapeHtml(L.heading)}" description="embedding">
      <p class="demo-hint">${escapeHtml(L.hint)}</p>
      <input class="pg-input pg-input-inline" id="pg-emb-in" type="text" spellcheck="false" value="${escapeHtml(L.inputPh)}" />
      <div class="pg-actions">
        <r-button type="primary" id="pg-emb-run">${escapeHtml(L.lookup)}</r-button>
      </div>
      <p class="pg-note">${escapeHtml(L.note)}</p>
      <div class="pg-block"><strong>${escapeHtml(L.table)}</strong>
        <div class="pg-emb-wrap"><table class="pg-emb-table" id="pg-emb-table"></table></div>
      </div>
      <div class="pg-block"><strong>${escapeHtml(L.shape)}</strong><pre class="demo-out" id="pg-emb-shape"></pre></div>
      <div class="pg-block"><strong>${escapeHtml(L.gathered)}</strong><pre class="demo-out" id="pg-emb-out"></pre></div>
    </r-card>
  `;

  const paint = () => {
    const raw = (host.querySelector('#pg-emb-in') as HTMLInputElement).value;
    const parsed = parseIds(raw, TEACH_EMB_VOCAB);
    const active = new Set(parsed.ids);
    const head =
      `<tr><th>id</th>` +
      Array.from({ length: TEACH_EMB_DIM }, (_, d) => `<th>d${d}</th>`).join('') +
      `</tr>`;
    const body = TEACH_EMB.map((row, i) => {
      const cls = active.has(i) ? ' class="pg-emb-hit"' : '';
      const cells = row.map((v) => `<td>${v.toFixed(2)}</td>`).join('');
      return `<tr${cls}><th>${i}</th>${cells}</tr>`;
    }).join('');
    host.querySelector('#pg-emb-table')!.innerHTML = head + body;

    const shapeEl = host.querySelector('#pg-emb-shape') as HTMLElement;
    const outEl = host.querySelector('#pg-emb-out') as HTMLElement;
    if (parsed.error === 'empty') {
      shapeEl.textContent =
        locale === 'cn' ? '请输入至少一个 id，例如 3, 7, 11' : 'Enter at least one id, e.g. 3, 7, 11';
      outEl.textContent = '';
      return;
    }
    if (parsed.error) {
      shapeEl.textContent =
        locale === 'cn'
          ? `无效 id「${parsed.error}」——教学表只有 0…${TEACH_EMB_VOCAB - 1}`
          : `Bad id "${parsed.error}" — teaching table only has 0…${TEACH_EMB_VOCAB - 1}`;
      outEl.textContent = '';
      return;
    }
    const ids = parsed.ids;
    const T = ids.length;
    shapeEl.textContent =
      locale === 'cn'
        ? `输入 ${T} 个编号 → 抄出 ${T} 行，每行 ${TEACH_EMB_DIM} 个小数`
        : `${T} ids in → copied ${T} rows, ${TEACH_EMB_DIM} floats each`;
    outEl.textContent = ids
      .map((id, t) => {
        const row = TEACH_EMB[id]!.map((v) => v.toFixed(2)).join(', ');
        return locale === 'cn'
          ? `第 ${t + 1} 个 token（id ${id}）抄到：[${row}]`
          : `token ${t + 1} (id ${id}) copied: [${row}]`;
      })
      .join('\n');
  };

  host.querySelector('#pg-emb-run')?.addEventListener('click', paint);
  host.querySelector('#pg-emb-in')?.addEventListener('keydown', (e) => {
    if ((e as KeyboardEvent).key === 'Enter') paint();
  });
  paint();
}

/** Tiny embed + linear next-token toy with SGD. */
function runTinyPretrain(lr: number, steps: number, seed: number): number[] {
  const V = 32;
  const D = 8;
  const T = 8;
  let s = seed >>> 0;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
  const randn = () => (rand() + rand() + rand() + rand() - 2) * 0.5;

  const emb = Array.from({ length: V * D }, () => randn() * 0.1);
  const w = Array.from({ length: D * V }, () => randn() * 0.1);
  const ids = Array.from({ length: T + 1 }, () => Math.floor(rand() * V));

  const losses: number[] = [];
  for (let step = 0; step < steps; step += 1) {
    let loss = 0;
    const gEmb = new Float64Array(V * D);
    const gW = new Float64Array(D * V);

    for (let t = 0; t < T; t += 1) {
      const xId = ids[t]!;
      const yId = ids[t + 1]!;
      const h = emb.slice(xId * D, xId * D + D);
      const logits = new Float64Array(V);
      for (let v = 0; v < V; v += 1) {
        let sum = 0;
        for (let d = 0; d < D; d += 1) sum += h[d]! * w[d * V + v]!;
        logits[v] = sum;
      }
      let max = -Infinity;
      for (let v = 0; v < V; v += 1) max = Math.max(max, logits[v]!);
      let Z = 0;
      const prob = new Float64Array(V);
      for (let v = 0; v < V; v += 1) {
        prob[v] = Math.exp(logits[v]! - max);
        Z += prob[v]!;
      }
      for (let v = 0; v < V; v += 1) prob[v]! /= Z;
      loss += -Math.log(Math.max(prob[yId]!, 1e-12));

      for (let v = 0; v < V; v += 1) {
        const dLogit = prob[v]! - (v === yId ? 1 : 0);
        for (let d = 0; d < D; d += 1) {
          gW[d * V + v]! += h[d]! * dLogit;
          gEmb[xId * D + d]! += w[d * V + v]! * dLogit;
        }
      }
    }
    loss /= T;
    losses.push(loss);
    for (let i = 0; i < emb.length; i += 1) emb[i]! -= (lr * gEmb[i]!) / T;
    for (let i = 0; i < w.length; i += 1) w[i]! -= (lr * gW[i]!) / T;
  }
  return losses;
}

function sparkline(values: number[]): string {
  if (values.length === 0) return '';
  const w = 280;
  const h = 64;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  const pts = values
    .map((v, i) => {
      const x = (i / Math.max(values.length - 1, 1)) * (w - 8) + 4;
      const y = h - 4 - ((v - min) / span) * (h - 8);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return `<svg class="pg-spark" viewBox="0 0 ${w} ${h}" width="100%" height="${h}" role="img"><polyline fill="none" stroke="currentColor" stroke-width="2" points="${pts}"/></svg>`;
}

export function mountPretrainPlayground(host: HTMLElement, locale: PlaygroundLocale): void {
  const L = LABELS[locale].pretrain;
  host.hidden = false;
  host.innerHTML = `
    <r-card heading="${escapeHtml(L.heading)}" description="pretrain">
      <p class="demo-hint">${escapeHtml(L.hint)}</p>
      <div class="pg-controls">
        <label class="pg-label">${escapeHtml(L.lr)}
          <input type="range" id="pg-lr" min="0.01" max="1" step="0.01" value="0.2" />
          <span id="pg-lr-val">0.20</span>
        </label>
        <label class="pg-label">${escapeHtml(L.steps)}
          <input type="range" id="pg-steps" min="5" max="80" step="1" value="30" />
          <span id="pg-steps-val">30</span>
        </label>
      </div>
      <div class="pg-actions">
        <r-button type="primary" id="pg-train-run">${escapeHtml(L.train)}</r-button>
      </div>
      <p class="pg-note">${escapeHtml(L.note)}</p>
      <div class="pg-block"><strong>${escapeHtml(L.lossTitle)}</strong><div id="pg-spark"></div></div>
      <pre class="demo-out" id="pg-train-out" aria-live="polite"></pre>
    </r-card>
  `;

  const lrEl = host.querySelector('#pg-lr') as HTMLInputElement;
  const stepsEl = host.querySelector('#pg-steps') as HTMLInputElement;
  const lrVal = host.querySelector('#pg-lr-val')!;
  const stepsVal = host.querySelector('#pg-steps-val')!;
  lrEl.addEventListener('input', () => {
    lrVal.textContent = Number(lrEl.value).toFixed(2);
  });
  stepsEl.addEventListener('input', () => {
    stepsVal.textContent = stepsEl.value;
  });

  const run = () => {
    const lr = Number(lrEl.value);
    const steps = Number(stepsEl.value);
    const losses = runTinyPretrain(lr, steps, 42);
    (host.querySelector('#pg-spark') as HTMLElement).innerHTML = sparkline(losses);
    const first = losses[0]!.toFixed(4);
    const last = losses[losses.length - 1]!.toFixed(4);
    const mid = losses[Math.floor(losses.length / 2)]!.toFixed(4);
    (host.querySelector('#pg-train-out') as HTMLElement).textContent =
      locale === 'cn'
        ? `步数 ${steps} · lr ${lr.toFixed(2)}\n首 ${first} → 中 ${mid} → 末 ${last}`
        : `steps ${steps} · lr ${lr.toFixed(2)}\nfirst ${first} → mid ${mid} → last ${last}`;
  };

  host.querySelector('#pg-train-run')?.addEventListener('click', run);
  run();
}
