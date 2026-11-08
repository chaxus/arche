/** Short source walkthroughs: real arche snippets + numbered callouts. */

export type Walkthrough = {
  file: string;
  lines: string[];
  callouts: { line: number; en: string; cn: string }[];
};

export const WALKTHROUGHS: Record<string, Walkthrough> = {
  embedding: {
    file: 'arche/embedding.py',
    lines: [
      'class TokenEmbedding(nn.Module):',
      '    def __init__(self, vocab_size: int, embed_dim: int):',
      '        super().__init__()',
      '        self.weight = nn.Embedding(vocab_size, embed_dim)',
      '',
      '    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:',
      '        return self.weight(input_ids)  # (B, T, D)',
    ],
    callouts: [
      { line: 3, en: 'A table with one row per token id.', cn: '一张表：每个 token id 对应一行向量。' },
      {
        line: 6,
        en: 'Lookup only — position comes later via RoPE, not here.',
        cn: '只做查表——位置稍后由 RoPE 提供，不在这里加。',
      },
    ],
  },
  gqa: {
    file: 'arche/attention.py',
    lines: [
      'q = self.q_proj(x)  # (B, T, n_heads * d)',
      'k = self.k_proj(x)  # fewer KV heads',
      'v = self.v_proj(x)',
      'q, k = apply_rope(q, k, ...)',
      'k, v = repeat_kv(k, v, n_rep)  # share KV across Q groups',
      'out = scaled_dot_product_attention(q, k, v, is_causal=True)',
    ],
    callouts: [
      { line: 1, en: 'Project to more query heads than KV heads.', cn: 'Query 头数可以多于 KV 头。' },
      {
        line: 4,
        en: 'repeat_kv expands KV so each query group has a matching key/value.',
        cn: 'repeat_kv 把 KV 扩开，让每组 query 都有对应的 K/V。',
      },
      { line: 5, en: 'Causal mask: token t only sees ≤ t.', cn: '因果掩码：位置 t 只能看见 ≤ t。' },
    ],
  },
  block: {
    file: 'arche/block.py',
    lines: [
      'def forward(self, x, cache=None, start_pos=0):',
      '    h, cache = self.attn(self.input_norm(x), cache, start_pos)',
      '    x = x + h',
      '    x = x + self.mlp(self.post_norm(x))',
      '    return x, cache',
    ],
    callouts: [
      {
        line: 1,
        en: 'Pre-norm: normalize before attention.',
        cn: 'Pre-norm：注意力前先归一化。',
      },
      { line: 2, en: 'Residual: add the attention delta back.', cn: '残差：把注意力增量加回去。' },
      { line: 3, en: 'Same pattern for the MLP branch.', cn: 'MLP 分支同一套路。' },
    ],
  },
  'kv-cache': {
    file: 'arche/cache.py',
    lines: [
      '@dataclass',
      'class KVCache:',
      '    k: torch.Tensor  # (B, kv_heads, T, d)',
      '    v: torch.Tensor',
      '',
      '    def append(self, k, v) -> "KVCache":',
      '        return KVCache(torch.cat([self.k, k], -2), torch.cat([self.v, v], -2))',
    ],
    callouts: [
      {
        line: 2,
        en: 'Cache stores keys/values along the sequence axis.',
        cn: '沿着序列维缓存 K/V。',
      },
      {
        line: 6,
        en: 'Decode appends one new step instead of recomputing the prefix.',
        cn: '解码时追加一步，而不是重算整段前缀。',
      },
    ],
  },
};

export function mountWalkthrough(
  host: HTMLElement,
  chapterId: string,
  localeId: string,
  heading: string,
): void {
  const data = WALKTHROUGHS[chapterId];
  if (!data) {
    host.hidden = true;
    host.replaceChildren();
    return;
  }
  host.hidden = false;
  const calloutText = (c: (typeof data.callouts)[number]) =>
    localeId === 'cn' ? c.cn : c.en;

  const linesHtml = data.lines
    .map((line, idx) => {
      const n = idx + 1;
      const active = data.callouts.some((c) => c.line === n);
      return `<div class="wt-line${active ? ' is-callout' : ''}" data-line="${n}"><span class="wt-n">${n}</span><code>${escapeHtml(line || ' ')}</code></div>`;
    })
    .join('');

  const notes = data.callouts
    .map((c) => `<li><strong>L${c.line}</strong> — ${escapeHtml(calloutText(c))}</li>`)
    .join('');

  host.innerHTML = `
    <r-card heading="${escapeHtml(heading)}" description="${escapeHtml(data.file)}">
      <div class="wt-code" role="region" aria-label="${escapeHtml(data.file)}">${linesHtml}</div>
      <ol class="wt-notes">${notes}</ol>
    </r-card>
  `;
}

function escapeHtml(s: string): string {
  return s
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}
