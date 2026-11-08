/** Inline chapter source: real repo files + optional line callouts. */

export type SourceCallout = { line: number; en: string; cn: string };

export type ChapterSource = {
  /** Path relative to repo root, e.g. arche/norms.py */
  library?: string;
  /** Highlighted teaching lines inside the library file */
  callouts?: SourceCallout[];
};

/** Map chapter id → library file readers should see first. */
export const CHAPTER_SOURCES: Record<string, ChapterSource> = {
  overview: {
    library: 'arche/config.py',
    callouts: [
      { line: 12, en: 'All sizes live in one frozen config.', cn: '尺寸都收在一个不可变配置里。' },
      { line: 37, en: 'llama32_1b() is the teaching target.', cn: '教学目标尺寸是 llama32_1b()。' },
    ],
  },
  tokenizer: {
    library: 'arche/tokenizer.py',
    callouts: [
      { line: 7, en: 'Official Llama 3.2 tokenizer id.', cn: '官方 Llama 3.2 tokenizer。' },
      { line: 17, en: 'Thin HF wrapper — we do not reinvent BPE.', cn: '薄封装；不重写 BPE。' },
    ],
  },
  embedding: {
    library: 'arche/embedding.py',
    callouts: [
      { line: 18, en: 'One row per token id.', cn: '每个 token id 对应一行向量。' },
      { line: 26, en: 'Lookup only — RoPE adds position later.', cn: '只做查表；位置由 RoPE 稍后加。' },
    ],
  },
  rmsnorm: {
    library: 'arche/norms.py',
    callouts: [
      { line: 14, en: 'Learnable scale γ; no bias.', cn: '可学习缩放 γ；无 bias。' },
      { line: 21, en: 'RMS over the last dim — no mean subtract.', cn: '最后一维做 RMS，不减均值。' },
      { line: 22, en: 'rsqrt(variance + eps) is the core scale.', cn: '核心就是 rsqrt(variance + eps)。' },
    ],
  },
  rope: {
    library: 'arche/rope.py',
    callouts: [
      { line: 77, en: 'rotate_half: pair dimensions and swap/negate.', cn: 'rotate_half：成对维度交换并取负。' },
      {
        line: 93,
        en: 'Apply: x·cos + rotate_half(x)·sin.',
        cn: '应用：x·cos + rotate_half(x)·sin。',
      },
    ],
  },
  gqa: {
    library: 'arche/attention.py',
    callouts: [
      { line: 40, en: 'More Q heads than KV heads (GQA).', cn: 'Q 头多于 KV 头（GQA）。' },
      { line: 85, en: 'repeat_interleave expands KV for each Q group.', cn: '把 KV 扩到与 Q 头对齐。' },
      { line: 90, en: 'Causal SDPA for the prefill pass.', cn: '预填充用因果注意力。' },
    ],
  },
  swiglu: {
    library: 'arche/mlp.py',
    callouts: [
      { line: 15, en: 'Three bias-free linears: gate / up / down.', cn: '三个无 bias 线性层。' },
      { line: 20, en: 'SwiGLU: silu(gate) * up, then down.', cn: 'SwiGLU：silu(gate)*up，再 down。' },
    ],
  },
  block: {
    library: 'arche/block.py',
    callouts: [
      { line: 41, en: 'Pre-norm before attention.', cn: '注意力前先 RMSNorm。' },
      { line: 42, en: 'Residual add after attention.', cn: '注意力后残差相加。' },
      { line: 43, en: 'Same pattern for the MLP branch.', cn: 'MLP 分支同一套路。' },
    ],
  },
  stack: {
    library: 'arche/model.py',
    callouts: [
      { line: 20, en: 'Embed tokens once at the bottom.', cn: '底部做一次 token 嵌入。' },
      { line: 24, en: 'Optional tie: LM head shares embed weight.', cn: '可选：LM head 与 embed 共享权重。' },
      { line: 46, en: 'Stack every TransformerBlock with its cache.', cn: '逐层跑 block，并带上各自 cache。' },
    ],
  },
  weights: { library: 'arche/load_weights.py' },
  generate: {
    library: 'arche/generate.py',
    callouts: [
      { line: 18, en: 'temperature ≤ 0 → greedy argmax.', cn: 'temperature ≤ 0 走贪心。' },
      { line: 30, en: 'generate() is the decode loop entry.', cn: 'generate() 是解码入口。' },
    ],
  },
  'kv-cache': {
    library: 'arche/cache.py',
    callouts: [
      { line: 14, en: 'K/V stored along the sequence axis.', cn: '沿序列维缓存 K/V。' },
      { line: 21, en: 'Decode appends one step — no prefix recompute.', cn: '解码追加一步，不重算前缀。' },
    ],
  },
  'rope-scaling': {
    library: 'arche/rope.py',
    callouts: [
      { line: 10, en: 'Llama-3 wavelength smoothing for long context.', cn: '长上下文用 Llama-3 波长平滑。' },
      { line: 33, en: 'Low-frequency bands get scaled down.', cn: '低频带缩小频率。' },
    ],
  },
  moe: { library: 'arche/moe.py' },
  mla: { library: 'arche/mla.py' },
  'device-quant-chat': { library: 'arche/chat.py' },
  'continuous-batching': { library: 'arche/batching.py' },
  pretrain: { library: 'arche/train/pretrain.py' },
  'pretrain-data': { library: 'arche/train/pretrain.py' },
  'pretrain-optim': { library: 'arche/train/pretrain.py' },
  'pretrain-scale': { library: 'arche/train/pretrain.py' },
  'sft-dpo': { library: 'arche/train/sft.py' },
  'frontier-ckpt': { library: 'arche/checkpoint.py' },
  'hf-align': { library: 'arche/align_hf.py' },
  'sharded-frontier': { library: 'arche/shards.py' },
  'paged-cuda-graph': { library: 'arche/paged_kv.py' },
};

// Eager raw imports from the monorepo (vite root = web/).
const RAW_CHAPTERS = import.meta.glob('../../../chapters/**/*.py', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const RAW_ARCHE = import.meta.glob('../../../arche/**/*.py', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

function resolveRaw(repoPath: string): string | null {
  const needle = repoPath.replace(/\\/g, '/');
  const pools = needle.startsWith('chapters/') ? RAW_CHAPTERS : RAW_ARCHE;
  for (const [key, text] of Object.entries(pools)) {
    const normalized = key.replace(/\\/g, '/');
    if (normalized.endsWith(`/${needle}`) || normalized.endsWith(needle)) {
      return text;
    }
  }
  return null;
}

export type SourceLabels = {
  heading: string;
  libraryTab: string;
  labTab: string;
  empty: string;
  github: string;
  copy: string;
  copied: string;
};

export type MountedSource = {
  calloutLines: number[];
  setActiveCallout: (line: number | null) => void;
};

export function mountSource(
  host: HTMLElement,
  chapterId: string,
  labScript: string,
  localeId: string,
  labels: SourceLabels,
): MountedSource | null {
  const meta = CHAPTER_SOURCES[chapterId];
  const libraryPath = meta?.library;
  const labPath = labScript || '';
  const libraryText = libraryPath ? resolveRaw(libraryPath) : null;
  const labText = labPath ? resolveRaw(labPath) : null;

  if (!libraryText && !labText) {
    host.hidden = true;
    host.replaceChildren();
    return null;
  }

  host.hidden = false;
  const defaultTab = libraryText ? 'library' : 'lab';
  const activePath = defaultTab === 'library' ? libraryPath! : labPath;
  const callouts = meta?.callouts ?? [];

  host.innerHTML = `
    <r-card heading="${escapeHtml(labels.heading)}">
      <div class="src-chrome">
        <div class="src-chrome-bar">
          <div class="src-tabs" role="tablist">
            ${
              libraryText
                ? `<button type="button" class="src-tab${defaultTab === 'library' ? ' is-active' : ''}" data-tab="library" role="tab">${escapeHtml(labels.libraryTab)}</button>`
                : ''
            }
            ${
              labText
                ? `<button type="button" class="src-tab${defaultTab === 'lab' ? ' is-active' : ''}" data-tab="lab" role="tab">${escapeHtml(labels.labTab)}</button>`
                : ''
            }
          </div>
          <span class="src-chrome-path" id="src-path">${escapeHtml(activePath)}</span>
          <button type="button" class="src-copy" id="src-copy">${escapeHtml(labels.copy)}</button>
        </div>
        <div class="src-body" id="src-body"></div>
      </div>
      <p class="src-github">
        <a href="https://github.com/chaxus/arche/blob/main/${escapeHtml(activePath)}" target="_blank" rel="noopener">${escapeHtml(labels.github)}</a>
      </p>
    </r-card>
  `;

  const body = host.querySelector('#src-body') as HTMLElement;
  const pathEl = host.querySelector('#src-path') as HTMLElement;
  const copyBtn = host.querySelector('#src-copy') as HTMLButtonElement;
  let currentText = '';

  const paint = (tab: 'library' | 'lab') => {
    const path = tab === 'library' ? libraryPath : labPath;
    const text = tab === 'library' ? libraryText : labText;
    if (!path || text == null) return;
    currentText = text;
    const showCallouts = tab === 'library' ? callouts : [];
    body.innerHTML = renderCode(text, showCallouts, localeId);
    pathEl.textContent = path;
    const link = host.querySelector('.src-github a') as HTMLAnchorElement | null;
    if (link) link.href = `https://github.com/chaxus/arche/blob/main/${path}`;
    host.querySelectorAll('.src-tab').forEach((btn) => {
      btn.classList.toggle('is-active', (btn as HTMLElement).dataset.tab === tab);
    });
    copyBtn.textContent = labels.copy;
  };

  paint(defaultTab === 'library' ? 'library' : 'lab');

  host.querySelectorAll('.src-tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      const tab = (btn as HTMLElement).dataset.tab as 'library' | 'lab';
      paint(tab);
    });
  });

  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(currentText);
      copyBtn.textContent = labels.copied;
      window.setTimeout(() => {
        copyBtn.textContent = labels.copy;
      }, 1400);
    } catch {
      copyBtn.textContent = labels.copy;
    }
  });

  const setActiveCallout = (line: number | null) => {
    body.querySelectorAll('.wt-line.is-callout').forEach((node) => {
      const el = node as HTMLElement;
      const n = Number(el.dataset.line);
      el.classList.toggle('is-active', line != null && n === line);
    });
    if (line == null) return;
    const target = body.querySelector(`.wt-line[data-line="${line}"]`) as HTMLElement | null;
    target?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  };

  return {
    calloutLines: callouts.map((c) => c.line),
    setActiveCallout,
  };
}

function renderCode(text: string, callouts: SourceCallout[], localeId: string): string {
  const lines = text.replace(/\n$/, '').split('\n');
  const calloutSet = new Set(callouts.map((c) => c.line));
  const linesHtml = lines
    .map((line, idx) => {
      const n = idx + 1;
      const active = calloutSet.has(n);
      return `<div class="wt-line${active ? ' is-callout' : ''}" data-line="${n}"><span class="wt-n">${n}</span><code>${escapeHtml(line || ' ')}</code></div>`;
    })
    .join('');

  const notes =
    callouts.length === 0
      ? ''
      : `<ol class="wt-notes">${callouts
          .map((c) => {
            const tip = localeId === 'cn' ? c.cn : c.en;
            return `<li><strong>L${c.line}</strong> — ${escapeHtml(tip)}</li>`;
          })
          .join('')}</ol>`;

  return `<div class="wt-code" role="region">${linesHtml}</div>${notes}`;
}

function escapeHtml(s: string): string {
  return s
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}
