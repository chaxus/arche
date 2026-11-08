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

/** Reading order: tables + next-token train + dense trunk → align → serve → frontier. */
export const PART_ORDER: PartId[] = ['start', 'trunk', 'train', 'infer', 'frontier'];

/** Stable sort: PART_ORDER first, then original list order within a part. */
export function chaptersInPartOrder(chapters: Chapter[]): Chapter[] {
  return chapters
    .map((chapter, index) => ({ chapter, index }))
    .sort((a, b) => {
      const partDelta =
        PART_ORDER.indexOf(a.chapter.partId) - PART_ORDER.indexOf(b.chapter.partId);
      return partDelta !== 0 ? partDelta : a.index - b.index;
    })
    .map(({ chapter }) => chapter);
}
