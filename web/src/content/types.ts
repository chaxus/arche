export type ChapterDemo = 'chat' | 'batch' | 'train';

export type Chapter = {
  id: string;
  ep: string;
  title: string;
  summary: string;
  markdown: string;
  /** Companion chapter script path relative to repo root. */
  script: string;
  latex?: string;
  demo?: ChapterDemo;
};
