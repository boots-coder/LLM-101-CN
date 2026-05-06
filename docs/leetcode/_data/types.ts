export type Difficulty = 'easy' | 'medium' | 'hard'

export type PatternSlug =
  | 'hashmap'
  | 'two-pointer'
  | 'sliding-window'
  | 'binary-search'
  | 'monotonic-stack'
  | 'heap'
  | 'linked-list'
  | 'tree-dfs'
  | 'tree-bfs'
  | 'trie'
  | 'backtracking'
  | 'graph'
  | 'dp-1d'
  | 'dp-2d'
  | 'bit-prefix'
  | 'python-ref'

export type MicroTag =
  | 'boundary'
  | 'invariant'
  | 'naming'
  | 'complexity'
  | 'data-structure'
  | 'syntax'
  | 'pythonism'

export interface MicroOption {
  id: string
  text: string
}

export interface MicroQuestion {
  id: string
  prompt: string
  codeContext?: string
  highlightLine?: number
  options: MicroOption[]
  answer: string
  explain: string
  pythonRefIds?: string[]
  tags?: MicroTag[]
}

export interface Problem {
  id: string
  leetcodeNo: number
  title: { zh: string; en: string }
  difficulty: Difficulty
  pattern: PatternSlug
  tags: string[]
  statement: string
  examples?: { input: string; output: string; note?: string }[]
  constraints?: string[]
  intuition: string
  solutionCode: string
  language: 'python'
  microQuestions: MicroQuestion[]
  complexity: { time: string; space: string }
  pythonRefIds?: string[]
}

export interface PatternMeta {
  slug: PatternSlug
  name: { zh: string; en: string }
  oneLiner: string
  day: 1 | 2 | 3 | 4 | 5
  problemIds: string[]
  bonusList: { no: number; title: string; difficulty: Difficulty }[]
}

export interface SrsCard {
  ease: number
  interval: number
  due: string
  reps: number
  lapses: number
}

export interface LcState {
  schemaVersion: 1
  xp: number
  hearts: { current: number; max: number; lastRefillAt: number }
  streak: { count: number; lastActiveDate: string; freezeTokens: number }
  dailyGoal: { xpTarget: number; xpToday: number; date: string }
  cards: Record<string, SrsCard>
  problemsCompleted: Record<string, { firstClearAt: number; bestAccuracy: number }>
  settings: { soundEnabled: boolean; heartsEnabled: boolean; reduceMotion: boolean }
}
