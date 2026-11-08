import type { LocaleDef } from './locales';

export type UiCopy = {
  brandTagline: string;
  chaptersNav: string;
  lessonHeading: string;
  companionPython: string;
  siteLabel: string;
  github: string;
  langMenu: string;
  progressLabel: string;
  demoHeading: string;
  demoChatHint: string;
  demoChatRun: string;
  demoBatchHint: string;
  demoBatchRun: string;
  demoTrainHint: string;
  demoTrainRun: string;
  markDone: string;
  markedDone: string;
};

const EN: UiCopy = {
  brandTagline: 'ἀρχή — modern dense from first principles',
  chaptersNav: 'Chapters',
  lessonHeading: 'Lesson',
  companionPython: 'Companion Python',
  siteLabel: 'Site',
  github: 'GitHub',
  langMenu: 'Language',
  progressLabel: 'Progress',
  demoHeading: 'Try it',
  demoChatHint: 'Render a tiny chat transcript the way ep16 falls back without a HF template.',
  demoChatRun: 'Render template',
  demoBatchHint: 'Three decode jobs share steps until each hits max_new_tokens — the ep17 scheduler shape.',
  demoBatchRun: 'Run scheduler',
  demoTrainHint: 'Toy pretrain then SFT/DPO: loss numbers are illustrative, not a real training run.',
  demoTrainRun: 'Simulate steps',
  markDone: 'Mark as read',
  markedDone: 'Read',
};

const CN: UiCopy = {
  brandTagline: 'ἀρχή — 从第一性原理手写现代 dense',
  chaptersNav: '章节',
  lessonHeading: '课程',
  companionPython: '配套 Python',
  siteLabel: '站点',
  github: 'GitHub',
  langMenu: '语言',
  progressLabel: '进度',
  demoHeading: '动手试',
  demoChatHint: '用 ep16 的回退模板渲染一小段对话（无需 HF chat_template）。',
  demoChatRun: '渲染模板',
  demoBatchHint: '三个解码任务共享 step，直到各自打满 max_new_tokens——对应 ep17 调度形状。',
  demoBatchRun: '跑调度器',
  demoTrainHint: '玩具 pretrain → SFT/DPO：数字仅示意，不是真实训练。',
  demoTrainRun: '模拟几步',
  markDone: '标记已读',
  markedDone: '已读',
};

export const uiFor = (locale: LocaleDef): UiCopy => (locale.id === 'cn' ? CN : EN);
