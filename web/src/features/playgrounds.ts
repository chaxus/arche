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

const LABELS: Record<PlaygroundLocale, { tokenizer: Labels; pretrain: Labels }> = {
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
