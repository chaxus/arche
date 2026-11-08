import type { LocaleDef } from '../i18n/locales';
import { CHAPTERS_CN } from './cn';
import { CHAPTERS_EN } from './en';
import type { Chapter } from './types';

export type { Chapter } from './types';

export const chaptersFor = (locale: LocaleDef): Chapter[] =>
  locale.id === 'cn' ? CHAPTERS_CN : CHAPTERS_EN;
