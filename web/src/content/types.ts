export type ChapterDemo =
  | 'chat'
  | 'batch'
  | 'train'
  | 'rope'
  | 'gqa'
  | 'kv'
  | 'tokenizer'
  | 'embedding'
  | 'pretrain';

export type PartId = 'start' | 'trunk' | 'infer' | 'train' | 'frontier';

export type Chapter = {
  id: string;
  ep: string;
  title: string;
  summary: string;
  markdown: string;
  /** Companion chapter script path relative to repo root; empty for site-only lessons. */
  script: string;
  partId: PartId;
  estimatedMinutes: number;
  latex?: string;
  demo?: ChapterDemo;
};

export const PART_ORDER: PartId[] = ['start', 'trunk', 'infer', 'train', 'frontier'];
