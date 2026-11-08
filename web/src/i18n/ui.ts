import type { LocaleDef } from './locales';

export type UiCopy = {
  brandTagline: string;
  chaptersNav: string;
  lessonHeading: string;
  companionPython: string;
  siteLabel: string;
  github: string;
  langMenu: string;
};

const EN: UiCopy = {
  brandTagline: 'ἀρχή — modern dense from first principles',
  chaptersNav: 'Chapters',
  lessonHeading: 'Lesson',
  companionPython: 'Companion Python',
  siteLabel: 'Site',
  github: 'GitHub',
  langMenu: 'Language',
};

const CN: UiCopy = {
  brandTagline: 'ἀρχή — 从第一性原理手写现代 dense',
  chaptersNav: '章节',
  lessonHeading: '课程',
  companionPython: '配套 Python',
  siteLabel: '站点',
  github: 'GitHub',
  langMenu: '语言',
};

export const uiFor = (locale: LocaleDef): UiCopy => (locale.id === 'cn' ? CN : EN);
