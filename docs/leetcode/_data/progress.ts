import type { LcState, SrsCard } from './types'

const STORAGE_KEY = 'llm101.lc.v1'
const HEART_REFILL_MS = 30 * 60 * 1000
const HEART_MAX = 5
const DEFAULT_DAILY_XP = 50

const isBrowser = typeof window !== 'undefined' && typeof localStorage !== 'undefined'

function todayStr(): string {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

function dateStrPlus(days: number): string {
  const d = new Date()
  d.setDate(d.getDate() + days)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

function defaultState(): LcState {
  return {
    schemaVersion: 1,
    xp: 0,
    hearts: { current: HEART_MAX, max: HEART_MAX, lastRefillAt: Date.now() },
    streak: { count: 0, lastActiveDate: '', freezeTokens: 0 },
    dailyGoal: { xpTarget: DEFAULT_DAILY_XP, xpToday: 0, date: todayStr() },
    cards: {},
    problemsCompleted: {},
    settings: { soundEnabled: true, heartsEnabled: true, reduceMotion: false },
  }
}

function migrate(raw: any): LcState {
  if (!raw || typeof raw !== 'object') return defaultState()
  const base = defaultState()
  return {
    ...base,
    ...raw,
    hearts: { ...base.hearts, ...(raw.hearts || {}) },
    streak: { ...base.streak, ...(raw.streak || {}) },
    dailyGoal: { ...base.dailyGoal, ...(raw.dailyGoal || {}) },
    cards: raw.cards || {},
    problemsCompleted: raw.problemsCompleted || {},
    settings: { ...base.settings, ...(raw.settings || {}) },
    schemaVersion: 1,
  }
}

let cache: LcState | null = null

export function loadState(): LcState {
  if (!isBrowser) return defaultState()
  if (cache) return cache
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    cache = raw ? migrate(JSON.parse(raw)) : defaultState()
  } catch {
    cache = defaultState()
  }
  return cache
}

export function saveState(state: LcState): void {
  if (!isBrowser) return
  cache = state
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {}
}

export function refillHearts(state: LcState): LcState {
  if (!state.settings.heartsEnabled) {
    state.hearts.current = state.hearts.max
    return state
  }
  const now = Date.now()
  const elapsed = now - state.hearts.lastRefillAt
  const refills = Math.floor(elapsed / HEART_REFILL_MS)
  if (refills > 0 && state.hearts.current < state.hearts.max) {
    state.hearts.current = Math.min(state.hearts.max, state.hearts.current + refills)
    state.hearts.lastRefillAt = state.hearts.lastRefillAt + refills * HEART_REFILL_MS
  }
  return state
}

export function loseHeart(state: LcState): LcState {
  if (!state.settings.heartsEnabled) return state
  if (state.hearts.current === state.hearts.max) {
    state.hearts.lastRefillAt = Date.now()
  }
  state.hearts.current = Math.max(0, state.hearts.current - 1)
  return state
}

export function addXp(state: LcState, amount: number): LcState {
  state.xp += amount
  if (state.dailyGoal.date !== todayStr()) {
    state.dailyGoal.date = todayStr()
    state.dailyGoal.xpToday = 0
  }
  state.dailyGoal.xpToday += amount
  return state
}

export function bumpStreak(state: LcState): LcState {
  const today = todayStr()
  if (state.streak.lastActiveDate === today) return state
  const yesterday = dateStrPlus(-1)
  if (state.streak.lastActiveDate === yesterday || !state.streak.lastActiveDate) {
    state.streak.count += 1
  } else {
    state.streak.count = 1
  }
  state.streak.lastActiveDate = today
  return state
}

function defaultCard(): SrsCard {
  return { ease: 2.5, interval: 0, due: todayStr(), reps: 0, lapses: 0 }
}

export function recordAnswer(state: LcState, cardId: string, grade: 1 | 2 | 3 | 4): LcState {
  const card = state.cards[cardId] ?? defaultCard()
  if (grade < 3) {
    card.lapses += 1
    card.interval = 1
    card.ease = Math.max(1.3, card.ease - 0.2)
  } else {
    card.reps += 1
    if (card.reps === 1) card.interval = 1
    else if (card.reps === 2) card.interval = 3
    else card.interval = Math.round(card.interval * card.ease)
    if (grade === 4) card.ease = Math.min(3.0, card.ease + 0.05)
  }
  card.due = dateStrPlus(card.interval)
  state.cards[cardId] = card
  return state
}

export function dueCards(state: LcState): string[] {
  const today = todayStr()
  return Object.entries(state.cards)
    .filter(([, c]) => c.due <= today)
    .map(([id]) => id)
}

export function markProblemCompleted(state: LcState, problemId: string, accuracy: number): LcState {
  const prev = state.problemsCompleted[problemId]
  state.problemsCompleted[problemId] = {
    firstClearAt: prev?.firstClearAt ?? Date.now(),
    bestAccuracy: Math.max(prev?.bestAccuracy ?? 0, accuracy),
  }
  return state
}

export function isProblemCompleted(state: LcState, problemId: string): boolean {
  return !!state.problemsCompleted[problemId]
}

export function resetAll(): void {
  if (!isBrowser) return
  cache = null
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {}
}
